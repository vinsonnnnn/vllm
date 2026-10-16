# 第四章：Scheduler 如何把连续请求变成每轮计算

固定 commit 与前章相同。以下主线使用同步 Scheduler、普通自回归、无 prefix cache、充足 KV；在此基础上单独讨论抢占与投机预算的边界。

## 1. 本章要回答的问题

1. 每轮为什么先看 running，再尝试 waiting？
2. 预算、请求数上限和 KV 容量各限制什么？
3. 长 prompt 怎样和已有请求的 decode 混在一轮？
4. 分配失败时，抢占到底撤销了什么？
5. 修改预算策略需要保持哪些输入输出契约？

## 2. 先建立直观理解

**连续批处理（Continuous Batching）**意味着在一轮轮计算边界重新决定参与者，而不是等整批请求全部结束再换下一批。每个请求本轮可以贡献一个 decode token，也可以贡献一段 prompt。

预算像一轮的工时额度，KV 像可用工作台。即使工时尚有剩余，没有工作台也不能接活；即使空闲 KV 很多，token 额度耗尽也必须等下一轮。**设计解释：**这两个约束分别控制计算规模和内存占用，不能只调大 token budget 就解决一切排队问题。

本版还区分**输入 token 预算**与**调度 token 预算**。在无投机的教学场景二者通常按相同的新计算位置扣减；投机中还要为 drafting 保留槽位，不能在优化时合并成一个随意使用的计数器。

## 3. 与 nano-vLLM 的对应关系

可以类比“从 waiting 加入可运行请求、为 running 安排计算、KV 不足时抢占”。不能直接类比为“有一个全局 prefill 阶段，然后切换到 decode 阶段”：这里 `Scheduler.schedule` 用 token 进度统一表达两类工作，具体分支仍会判断 prefill 特征以处理分块和其他限制。

## 4. 源码阅读地图

| 顺序 | 位置与符号 | 阅读目的 |
| --- | --- | --- |
| 1 | [vllm/v1/core/sched/scheduler.py:535](../../vllm/v1/core/sched/scheduler.py#L535)，`schedule` | 先读开头的统一进度说明与局部预算初始化 |
| 2 | [vllm/v1/core/sched/scheduler.py:586](../../vllm/v1/core/sched/scheduler.py#L586)，running 循环 | 找需求量、预算裁剪和 KV 分配顺序 |
| 3 | [vllm/v1/core/sched/scheduler.py:812](../../vllm/v1/core/sched/scheduler.py#L812)，waiting 循环 | 看请求数限制、缓存查询和加入条件 |
| 4 | [vllm/v1/core/sched/scheduler.py:988](../../vllm/v1/core/sched/scheduler.py#L988)，waiting 的 token 需求 | 恢复请求也可能有已生成 token，需要重算它们 |
| 5 | [vllm/v1/core/sched/request_queue.py:75](../../vllm/v1/core/sched/request_queue.py#L75)，`FCFSRequestQueue` | 先到先服务队列的实际操作，区别策略与数据结构 |
| 6 | [vllm/v1/core/sched/scheduler.py:1440](../../vllm/v1/core/sched/scheduler.py#L1440)，`_preempt_request` | 对照释放 KV、归零进度、保留内容和回队列 |
| 7 | [vllm/v1/core/sched/scheduler.py:1483](../../vllm/v1/core/sched/scheduler.py#L1483)，`_update_after_schedule` | 确认预记账和计划快照的先后 |
| 8 | [vllm/v1/core/sched/output.py:219](../../vllm/v1/core/sched/output.py#L219)，`SchedulerOutput` | 检查策略输出能否被 Runner 正确消费 |

## 5. 一轮真实过程

核心 `EngineCore.step` **普通调用** Scheduler；Scheduler 的一次调用大致按以下因果关系执行，不涉及 ZMQ 或 GPU kernel：

1. 初始化本轮集合、token_budget、input_budget 等局部状态。
2. 遍历 running，按已知 token、候选、placeholder 和 computed 算待做工作；裁剪到预算与长度限制。
3. 为该工作请求 KV；若不足，按策略选择抢占对象并重试。有些请求本轮无法推进时可以继续看其他请求，不等于严格全局 FCFS。
4. 将可执行的 running 请求加入计划，扣减额度；还有余量且满足条件时，处理 waiting／skipped_waiting。
5. 新请求先查可复用前缀，确定实际待算位置；按分块设置、预算、KV 和其他限制决定可否加入，状态改为 RUNNING。
6. 打包 SchedulerOutput，调用调度后更新。之后**核心→uni 普通调用→GPU 工作**才发生；结果回收后另调 `update_from_output`。

这不是事务性的纯计划函数：预算是局部变量，但请求进度、队列、KV 所有权都可能改变。修改中途的 continue／break 或分配顺序，可能同时改变公平性和资源生命周期。

## 6. 关键源码精读

### 6.1 running 的需求量与预算

<!-- source: vllm/v1/core/sched/scheduler.py:619-628 -->

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
num_new_tokens = min(
    num_new_tokens, token_budget, input_budget - draft_slots
)
```

输入来自 Request 的三类进度与本轮预算，输出是待进一步检查的 `num_new_tokens`。这里尚未确认 KV 分配成功，也还会受模型长度等限制。无投机且同步时，spec 与 placeholders 为零，退化为“已知 token 总数减去已安排计算进度”。新选出的一个 token 使下一轮出现一个待算位置。

**设计解释：**统一差值允许分块 prompt、decode 和投机候选通过同一接口表示；但这不意味着它们执行成本相同。把每个 token 当等成本，是调度额度的一种近似，不是延迟预测模型。

### 6.2 已有请求扣减两类预算

<!-- source: vllm/v1/core/sched/scheduler.py:758-763 -->

```python
request_id = request.request_id
req_to_new_blocks[request_id] = new_blocks
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
input_budget -= num_new_tokens + draft_slots
req_index += 1
```

输入是已经通过分配检查的请求；输出写入本轮计划，局部预算被消费。`draft_slots` 来自 speculative config 的 drafting 槽位需求，不应直接等同为“这轮实际验证候选数”。waiting 路径也有对应扣减，见 [vllm/v1/core/sched/scheduler.py:1214](../../vllm/v1/core/sched/scheduler.py#L1214)。修改其中一路而漏掉另一路会破坏额度一致性。

### 6.3 抢占重置计算进度，不删除输出内容

<!-- source: vllm/v1/core/sched/scheduler.py:1456-1462 -->

```python
self._free_request_blocks(request)
self.encoder_cache_manager.free(request)
self._inflight_prefills.discard(request)
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
if request.spec_token_ids:
    request.spec_token_ids = []
```

输入是被选中的 RUNNING 请求；状态变 PREEMPTED，KV 释放或安全延迟释放，计算进度归零，旧投机候选清掉。prompt 与已提交 output 不在这里清空，后续可以重算。方法后半段处理 in-flight／stale 信息并回等待队列。异步下不能仅照抄这几行完成抢占，完整协议见第七章。

## 7. 两请求贯穿案例

**教学示例，非日志。**A：prompt 长 4，输出上限 3；B：prompt 长 6，输出上限 2。A 先入队。每轮 token_budget=input_budget=4，max_num_seqs=2，启用 chunked prefill、不额外限制长 prefill，KV 充足，无 EOS／prefix／投机。每请求一个教学块即可容纳本例内容。

| 轮次 | running 优先安排 | waiting／续算安排 | 本轮回收 | computed 与输出长度（回收后） |
| --- | --- | --- | --- | --- |
| 1 | 开始无 running | A 的 prompt 4，预算用完，B 等待 | A→55 | A：4／1；B：0／0 |
| 2 | A 输入 55，用 1 | B prompt 的前 3，用 3 | A→56；B 无输出 | A：5／2；B：3／0 |
| 3 | A 输入 56，用 1；B 余下 prompt 3 | 无额度再接其他工作 | A→57 结束；B→77 | A：6／3；B：6／1 |
| 4 | B 输入 77，用 1 | 无其他请求 | B→88 结束 | B：7／2 |
| 清理 | 消费完成 ID 所需工作 | 不必产生用户 token | Worker 清理持久行 | 核心与 Worker 都收尾后才能视为空闲 |

第二轮混合了 decode 与 prefill；第三轮 B 已是 running，却仍在处理 prompt。可见“running 请求”不等于“每轮都只算一个 decode token”。第二轮 B 的部分 prompt 产生 hidden states，但不应提前向用户输出正式 completion token。

反事实一：max_num_seqs=1，A 结束前 B 无法加入，即使额度有剩余。反事实二：第二轮没有可分配 KV，B 不能因为预算剩 3 就强行进入。反事实三：关闭 chunked prefill 后，需要重新查 waiting 的完整输入可容纳规则，不能继续照抄表格。

## 8. 动手实验

**已验证**：本章源码引用和摘录。数值表为手算。**仅提供步骤**：依赖齐全后运行已有 Scheduler 测试，不需要先启动大模型。

```bash
uv run --no-project --offline --python .venv/bin/python python -m pytest tests/v1/core/test_scheduler.py -q -k 'schedule or preempt'
```

前置条件：pytest、Torch 与测试 fixture 依赖；本机最初缺少这些包，因此不宣称执行成功。先观察测试如何造请求、如何配置 token 限制，再挑对应测试名，不新写一个镜像实现的测试套件。

无 GPU 替代：把教学预算改为 3，重新算前两轮，并解释 A 的 prompt 分块为什么第一轮不应输出正式 token。用以下检索检查自己的假设：

```powershell
rg -n 'token_budget|input_budget|enable_chunked_prefill|max_num_running_reqs' vllm/v1/core/sched/scheduler.py
rg -n 'num_new_tokens == 0|preempted_req|_update_after_schedule' vllm/v1/core/sched/scheduler.py
```

观察点不是“打印了哪些文件”，而是每一轮计划总量、状态修改和分配条件是否能说明原因。

## 9. 易错点与自我检验

1. running 请求 B 为什么还能一次安排三个 prompt token？
2. `sum(num_scheduled_tokens.values())` 能代表本轮返回多少 token 吗？
3. input_budget 与 token_budget 在何种场景不能混用？
4. 额度充足但请求不被接纳，至少还可能缺什么？
5. 抢占时清空 output token 是否可行？
6. 调大预算为什么不保证 TTFT、TPOT 和吞吐同时改善？
7. 新策略为什么必须同时检查 running 与 waiting 的分支？

## 10. 本章总结

现在应能按源码手算混合 prefill/decode 的一轮计划，解释三类约束、调度预记账和抢占重算。还没有证明任何新策略更快，也没有分析真实负载下的公平性。

下一章追踪 Scheduler 所请求的 KV 如何落到物理块与槽位：预算回答“做多少”，资源映射回答“写到哪里、读到哪里、何时能复用”。

### 自我检验答案

1. running 只表示已接纳并保留进度，不表示 prompt 已全部完成。
2. 不能，它统计计算位置；分块 prefill 可能无输出，投机一请求可能输出多个。
3. 投机 drafting 槽位保留等情况下，扣减与上限的含义不同。
4. 请求槽位、KV 容量、长度约束、被阻塞状态或其他支持条件；逐分支查证。
5. 不行，已提交输出是恢复重算和用户连续性的依据。
6. 批次规模改变排队、单轮耗时和资源占用；需要按负载分别测量。
7. 两路都有额度、分配和状态更新，修改一处不足以保证全局契约成立。
