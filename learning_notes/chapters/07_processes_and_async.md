# 第七章：进程通信与异步执行的状态账本

固定版本不变。本章先保持单 GPU、uni、MRV2，再把 `async_scheduling` 从 False 恢复为 True；不同时打开分布式和投机。第五章的资源安全在本章落实到在飞行批次。

## 1. 本章要回答的问题

1. 前端进程、核心线程、执行器和 CUDA stream 是几种不同边界？
2. CPU 还没取回上轮 token，下一轮如何被调度？
3. placeholder、in-flight 与实际 token 如何对账？
4. Future 何时真正等待设备结果？
5. 取消、抢占和 KV 复用怎样避免晚到结果破坏新状态？

## 2. 先建立直观理解

**异步调度**让 CPU 在设备执行当前工作时准备后续计划。它不是允许模型跳过自回归依赖：下一 token 的值仍来自前一轮。区别是设备上的最近采样值可供下一轮读取，CPU 不必先把它转成 Python 整数再安排计划。

**占位输出（placeholder）**表示 CPU 知道某轮将产生输出位置，但还没消费其实际值；**在飞行 token（in-flight）**表示已安排但结果未回收的位置数；**CUDA stream**是设备工作依赖和顺序的载体。它们各自解决不同问题，不是三个名字描述同一个队列。

## 3. 与 nano-vLLM 的对应关系

同步小引擎可以假设“step 返回后状态全部已知”。这里必须区分已经安排、GPU 已经执行、D2H 已经完成、Scheduler 已经消费四个时刻。仍然使用已有 Scheduler 和 Runner 概念，只是它们之间多了可重叠的工作与对账协议。

## 4. 源码阅读地图

| 顺序 | 位置与符号 | 本章重点 |
| --- | --- | --- |
| 1 | [vllm/v1/engine/core_client.py:976](../../vllm/v1/engine/core_client.py#L976)，`SyncMPClient.add_request`；第 944 行 `get_output` | 同步用户 API 也能连接异步运行的核心 |
| 2 | [vllm/v1/engine/core.py:1705](../../vllm/v1/engine/core.py#L1705)，`process_input_sockets`；第 1808 行 `process_output_sockets` | socket 线程与核心主循环的队列交接 |
| 3 | [vllm/config/vllm.py:567](../../vllm/config/vllm.py#L567)，`max_concurrent_batches` | 单 GPU MRV2 开启异步时，批次容量由 1 变 2 |
| 4 | [vllm/v1/engine/core.py:649](../../vllm/v1/engine/core.py#L649)，`step_with_batch_queue` | 先提交工作，再按队列回收旧结果 |
| 5 | [vllm/v1/core/sched/async_scheduler.py:19](../../vllm/v1/core/sched/async_scheduler.py#L19)，`_update_after_schedule`；第 51 行 `_update_request_with_output` | 占位数创建与结算 |
| 6 | [vllm/v1/worker/gpu/async_utils.py:115](../../vllm/v1/worker/gpu/async_utils.py#L115)，`AsyncOutput` | main/copy stream、保留张量引用与事件同步 |
| 7 | [vllm/v1/executor/uniproc_executor.py:32](../../vllm/v1/executor/uniproc_executor.py#L32)，`AsyncOutputFuture` | Future.result 如何接到 AsyncOutput.get_output |
| 8 | [vllm/v1/core/sched/scheduler.py:2557](../../vllm/v1/core/sched/scheduler.py#L2557)，`_free_request_blocks`；第 2583 行 `_drain_deferred_frees` | 有条件延迟回收及安全栅栏 |

## 5. 边界图与真实调用链

| 边界 | 当前主线的机制 | 不能误解为 |
| --- | --- | --- |
| 前端→核心进程 | ZMQ ADD／ABORT／输出消息 | Python 跨进程直接调用对象方法 |
| 核心输入线程→核心主循环 | input_queue 交接已预处理对象 | 每次都经过共享内存广播 |
| 核心→uni Worker | 本进程方法分发 | 新增 GPU Worker 子进程 |
| CPU→GPU | PyTorch／Triton 发起工作 | Python 函数返回就表示 GPU 全部完成 |
| main stream→copy stream | stream 等待、异步 D2H、copy_event | 可以不保留源张量生命周期 |
| 批次结果→Scheduler | batch queue 持有匹配的计划与 Future | 按当前最新请求顺序消费任意旧结果 |

第一个进程边界在第一章已有；这里不要因为开启 AsyncScheduler 就把离线接口替换成 AsyncLLM。前端是否使用 asyncio、核心是否异步调度、执行器是否多进程是独立维度。

多进程执行器属于扩展路标：`MultiprocExecutor` 使用 `MessageQueue` 广播执行任务并取回响应，见 [vllm/v1/executor/multiproc_executor.py:420](../../vllm/v1/executor/multiproc_executor.py#L420)。这条路径会用共享内存通信组件；它不是当前 uni 链路，也不是前端 ZMQ 的另一名称。

## 6. 关键源码精读

### 6.1 保留计划和 Future 成对入队

`EngineCore.step_with_batch_queue`：

<!-- source: vllm/v1/engine/core.py:703-711 -->

```python
if not deferred_scheduler_output:
    # Add this step's future to the queue.
    batch_queue.appendleft((future, scheduler_output, exec_future))
    if len(batch_queue) < self.batch_queue_size and (
        model_executed or self.scheduler.has_requests()
    ):
        # Don't block on next worker response unless the queue is full
        # or there are no more requests to schedule.
        return None, model_executed
```

输入是已发起工作对应的 Future、计划与执行 Future；状态变化是队列新增一个批次。空间尚有余量时返回 None，允许继续提交，不立即阻塞等待结果。后面用 `pop()` 取最早批次，取得结果后把同一份 SchedulerOutput 传给 `update_from_output`。**设计解释：**计划快照与结果绑定，才能正确撤销拒绝位置、对账 token 和处理晚到输出。

### 6.2 输出值未知时先登记占位

`AsyncScheduler._update_after_schedule`：

<!-- source: vllm/v1/core/sched/async_scheduler.py:38-44 -->

```python
cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
request.num_output_placeholders += (
    self.num_sampled_tokens_per_step + cur_num_spec_tokens
)
# Add placeholders for the new draft/spec tokens.
# We will update the actual spec token ids in the worker process.
request.spec_token_ids = self._spec_token_placeholders
```

输入是本轮计划；输出是请求下一轮可用的占位进度。普通 AR 路径每轮基本采样数为 1，无投机时不需要候选占位。部分 prompt chunk 不生成输出，会在前面的条件跳过。真实 token 留在 Worker 设备状态中，第六章的 `combine_sampled_and_draft_tokens` 可直接从中组装下一轮输入。

### 6.3 转 Python 列表前等待复制完成

`AsyncOutput.get_output`：

<!-- source: vllm/v1/worker/gpu/async_utils.py:167-178 -->

```python
def get_output(self) -> ModelRunnerOutput:
    self.copy_event.synchronize()

    # NOTE(woosuk): The following code is to ensure compatibility with
    # the existing model runner.
    # Going forward, we should keep the data structures as NumPy arrays
    # rather than Python lists.
    sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
    num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
    for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
        del token_ids[num_tokens:]
    self.model_runner_output.sampled_token_ids = sampled_token_ids
```

输入是此前发起的异步输出复制；输出为 CPU 可消费的 ModelRunnerOutput。它等待 copy_event，然后按有效生成数裁掉无效位置。构造阶段通过 copy_stream 等待 main_stream，并保留 GPU 张量引用；否则复制完成前源对象可能失去生命周期保障。

### 取消与延迟回收的适用条件

结果更新会跳过不存在／已完成请求；抢占又单独维护 stale 输出。不能简单把所有晚到输出当成新生成内容。另一方面，不要断言本版所有异步场景都开启 deferred free：`defer_block_free` 初始 False，当前初始化只在存在 KV consumer connector 且允许多批次在飞行时开启，见 [vllm/v1/core/sched/scheduler.py:147](../../vllm/v1/core/sched/scheduler.py#L147) 至第 172 行。

**源码事实：**该保护避免 connector 的装载写入与旧批次 GPU 写入缺乏顺序约束时发生冲突；普通同一执行流上的复用与这一跨路径风险不同。**设计解释：**安全释放必须依据实际写入依赖，不应盲目全局延迟，也不应为了降低占用删除必要栅栏。

## 7. 两轮在飞行的账本

**教学示例，非 trace。**R 已生成 55，prompt 长 4，已有 computed=4、all 长 5；现在允许两个普通 decode 批次在飞行。忽略终止边界，单 GPU PP=1。

| CPU 事件 | computed | in-flight | output placeholders | CPU 已知 all 长度 |
| --- | --- | --- | --- | --- |
| 调度 S1，输入 55 | 5 | 1 | 1 | 5 |
| 未消费 S1，调度 S2 | 6 | 2 | 2 | 5 |
| 回收 S1 的 56 | 6 | 1 | 1 | 6 |
| 回收 S2 的 57 | 6 | 0 | 0 | 7 |

S2 的真实输入依赖设备先算出的 56；CPU 的占位只允许表达“存在这个位置”，不会把 -1 当作有效模型 token。GPU 同一依赖链仍按顺序推进。上述表仅是合法交错的一种，不保证某次运行一定在回收前提交满两批。

若 S1 回收后发现 R 已结束，S2 的晚到结果不能重新创建它。若抢占重置了 computed／placeholders，旧结果不能再次按正常路径扣减这些已归零计数；`is_stale` 分支就是需要连同预记账一起检查的协议。

## 8. 动手实验

**已验证**：本章分支与摘录。**仅提供步骤**：依赖齐全后运行 AsyncScheduler 测试：

```bash
uv run --no-project --offline --python .venv/bin/python python -m pytest tests/v1/core/test_async_scheduler.py -q
```

观察 placeholders、stale、终止和抢占对应的已有断言。GPU 环境建立后，创建两个独立配置，只改变 async_scheduling，保持模型／输入／采样／图模式一致；先比较有效输出，再用单独 Nsight 运行观察 CPU 准备与 GPU 工作的重叠。不要把 profiler 下的耗时当正式吞吐结果。

无 GPU 替代：按第七节账本插入一次抢占，逐个标记 S1/S2 哪个结果属于旧状态；查 `drop_stale_output` 和 `num_stale_output_tokens`。预期能说明“丢弃结果”和“释放块”是两个决策，而不是只说 async 更快。

## 9. 易错点与自我检验

1. 同步 `LLM.generate` 能否使用异步 Scheduler？
2. 开启异步调度是否意味着两个有依赖的 forward 可任意并行？
3. placeholder 与实际 spec token ID 是同一份内容吗？
4. 为什么 batch queue 保存 SchedulerOutput？
5. `.tolist()` 前的事件同步保证了什么？
6. deferred free 是否对所有 async 请求默认开启？
7. 抢占后的 stale 输出为什么不能正常扣减 placeholder？

## 10. 本章总结

现在应能画出进程、线程、Future、CUDA stream 和批次队列的不同边界，跟踪占位与真实输出如何对账，并定位取消和资源释放的条件。没有覆盖多节点、TP／PP 细节与 connector 协议全貌。

下一章减少重复发起 GPU 工作的 CPU 开销：把已整理的输入放进固定形状缓冲区，理解 CUDA Graph 的匹配、padding 和回放，并保留这里的依赖顺序。

### 自我检验答案

1. 能，用户 API 的阻塞方式与核心内部调度模式独立。
2. 不意味着，自回归值依赖和设备顺序仍必须满足；主要重叠 CPU 准备与设备工作。
3. 不是，CPU 占位表达未来位置，Worker 使用实际设备 token。
4. 结果必须与生成它的计划对账，尤其是不同批次的 token 数、候选和请求状态。
5. CPU 读取的复制目标已完成对应 D2H；不自动证明未来设备工作也完成。
6. 不是，本版初始化有 KV consumer connector 与多批次条件。
7. 抢占可能已重置计数，再扣会下溢或破坏新一轮状态；要按 stale 协议处理。
