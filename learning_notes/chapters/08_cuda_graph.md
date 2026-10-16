# 第八章：CUDA Graph、动态批次与 padding

固定 commit 沿用首页。本章在第六章 MRV2 输入路径上讨论图执行，明确区分运行模式与用户配置模式；不从旧 Runner 的 dispatcher 推断当前行为。

## 1. 本章要回答的问题

1. eager、piecewise 和 full 分别改变哪一段执行？
2. 动态请求集合如何匹配已 capture 的图？
3. 有效 token 数和 padded token 数分别由谁使用？
4. 为什么逻辑上减少 token，未必带来同比例 GPU 时间减少？

## 2. 先建立直观理解

**capture（捕获）**先记录一组设备操作；**replay（回放）**在后续复用执行结构。减少 CPU 重复发起的开销，需要输入缓冲区、形状和上下文满足约束，不是把一个 Python 函数的返回值缓存起来。

实际请求数和 token 数不断变化。**padding（填充）**让某些批次使用较大的已支持形状，多出来的位置必须不产生有效输出和错误 KV 写入。它可能省 launch 开销，也可能多做一些设备工作，必须通过负载测量判断取舍。

**动态形状编译**与 CUDA Graph 也不是同一机制：前者决定编译代码对形状的适用方式，后者重放捕获的设备工作。当前配置会协调二者；不能仅看到 torch.compile 就认定一定是 FULL graph。

## 3. 与 nano-vLLM 的对应关系

可以类比“捕获若干 batch size，再对齐选择”。当前 MRV2 的匹配条件不只是 batch size：还包含 token 数、请求数限制、uniform token count、query 长度、LoRA 与 microbatch 等。FULL 不应被讲成永远只支持每请求一个 token；本版存在更丰富的匹配路径，要看描述符和所选能力。

## 4. 源码阅读地图

| 顺序 | 位置与符号 | 要看什么 |
| --- | --- | --- |
| 1 | [vllm/config/compilation.py:53](../../vllm/config/compilation.py#L53)，`CUDAGraphMode` | NONE／PIECEWISE／FULL 与组合配置的区别 |
| 2 | [vllm/v1/worker/gpu/cudagraph_utils.py:59](../../vllm/v1/worker/gpu/cudagraph_utils.py#L59)，`BatchExecutionDescriptor` | 一次执行的形状和模式协议 |
| 3 | [vllm/v1/worker/gpu/cudagraph_utils.py:98](../../vllm/v1/worker/gpu/cudagraph_utils.py#L98)，`_is_compatible` | 图描述与实际批次的完整匹配条件 |
| 4 | [vllm/v1/worker/gpu/cudagraph_utils.py:343](../../vllm/v1/worker/gpu/cudagraph_utils.py#L343)，`CudaGraphManager.capture`；第 435 行 `dispatch` | capture 与 runtime dispatch 是不同阶段 |
| 5 | [vllm/v1/worker/gpu/model_runner.py:1617](../../vllm/v1/worker/gpu/model_runner.py#L1617)，图 dispatch 调用；第 1202 行 `prepare_inputs` | 选择后才按描述准备本轮缓冲区 |
| 6 | [vllm/v1/worker/gpu/model_runner.py:1803](../../vllm/v1/worker/gpu/model_runner.py#L1803)，模型运行分支 | FULL 回放／PIECEWISE／eager 在真实执行处汇合 |
| 7 | [vllm/v1/worker/gpu/cudagraph_utils.py:468](../../vllm/v1/worker/gpu/cudagraph_utils.py#L468)，`run_fullgraph`；第 658 行子类输出处理 | 找 replay 与稳定输出缓冲区 |

旧路径路标：`vllm/v1/cudagraph_dispatcher.py` 也存在，但本章主链调用 MRV2 的 CudaGraphManager；不能把两者同名 dispatch 混为一个调用。

## 5. 真实链路

初始化阶段的 **CPU 控制→GPU capture** 与每轮 runtime 的流程要分开。模型和缓存初始化后，MRV2 `capture_model` 组织捕获；本轮 `execute_model` 则先汇聚请求状态、计算真实 token 数，再 `dispatch_cg_and_sync_dp` 选择描述符。

随后 **CPU／GPU 输入准备**填入已持有的缓冲区，padding 的请求边界、seq_lens 和槽位按规则设无效。运行阶段按 `batch_desc.cg_mode`：FULL 调 `run_fullgraph`，PIECEWISE 调 `run_pw_graph`，NONE 调模型。FULL 回放并不重新走一遍普通 Python 模型调用。

**源码事实：**第 1804 行的注释说明 FULL 回放时不需要重新传入输入张量，因为已复制到图输入缓冲区。回放输出仍需进入第六、七章的采样、D2H 和状态更新链；图执行不会绕过请求生命周期。

## 6. 关键源码精读

### 6.1 匹配失败有明确的 eager 描述

`CudaGraphManager.dispatch` 在遍历捕获候选后：

<!-- source: vllm/v1/worker/gpu/cudagraph_utils.py:460-466 -->

```python
return BatchExecutionDescriptor(
    cg_mode=CUDAGraphMode.NONE,
    num_tokens=num_tokens,
    num_reqs=num_reqs,
    num_active_loras=effective_loras,
    num_ubatches=num_ubatches,
)
```

输入是实际批次大小、query 特征和其他描述；无匹配时返回 NONE，保留实际尺寸。它不偷偷把任意批次塞进最近的图。上面的 `_is_compatible` 同时检查描述符是否覆盖 token／请求规模和其他约束；因而“找最近 bucket”必须以合法性为前提。

### 6.2 输入有效长度与承载长度

`GPUModelRunner.prepare_inputs`：

<!-- source: vllm/v1/worker/gpu/model_runner.py:1208-1210 -->

```python
num_tokens = batch_req_state.num_tokens
num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)
assert num_tokens > 0
```

实际工作量来自 batch_req_state，执行承载长度还要覆盖描述符的尺寸。后续 InputBatch 同时保留二者，供不同准备与执行环节消费。这里不增加用户请求的实际输出长度，也不应该把 padding 计入有效吞吐的输出 token 分母。

### 6.3 回放前仍要满足流依赖

`CudaGraphManager.run_fullgraph`：

<!-- source: vllm/v1/worker/gpu/cudagraph_utils.py:480-481 -->

```python
get_offloader().sync_prev_onload()
self.graphs[desc].replay()
```

输入是已验证、已捕获的描述符；输出是图对稳定缓冲区的更新。前面的注释解释：此前 eager／piecewise 的异步装载可能位于另一 stream，图捕获时的事件未必能看到这些未来复制，所以先处理 onload 顺序。**设计解释：**回放省去重复发起，不取消数据依赖；删除同步前必须证明仍然保序。

## 7. 从有效 token 到图形状的案例

**教学示例，非当前默认 capture sizes。**假设有一个合法描述符承载 8 个 token，实际批次 7 个，其他请求数和 query 条件均兼容。

| 场景 | 有效 token | 承载 token | 应如何解释 |
| --- | --- | --- | --- |
| eager | 7 | 7 | 直接计算当前有效形状 |
| 匹配图 | 7 | 8 | 一个填充位置不应贡献新请求／输出／有效 KV |
| 候选裁剪后仍匹配同图 | 6 | 8 | 有效工作减少，但部分 kernel 的承载规模不变 |
| 其他匹配条件失败 | 7 | 7 的 NONE 描述 | 不应强行使用上面 8-token 图 |

最后一行不能只靠 token 数推断：例如 max_query_len 超过捕获约束就可能不兼容。第三行也不能简单说“一点都不会省”：变长 attention 可能按 metadata 少做部分工作，但 padded dense 层可能保持同样规模。只有 trace 与端到端实验能说明总收益。

## 8. 动手实验

**已验证**：MRV2 分支、描述符字段和摘录。**仅提供步骤**：具备依赖后运行已有图测试：

```bash
uv run --no-project --offline --python .venv/bin/python python -m pytest tests/v1/cudagraph/test_cudagraph_manager.py -q
```

先阅读测试选用的 manager 类型，区分纯 dispatch 测试与需要 CUDA capture 的部分。本轮未执行。

GPU 环境就绪后，用独立实例对比 eager 与合法 graph 配置，固定请求、采样、异步开关和资源；先核对输出 token，再单独记录初始化／capture 成本与 warmup 后的生成时长。第一章脚本强制 eager，不能运行原脚本就声称已经测到 graph。

无 GPU 替代：给第七节描述符补上 num_reqs、uniform_token_count、max_query_len，逐项对照 `_is_compatible` 判断合法性。以下只读命令可执行：

```powershell
rg -n 'def _is_compatible|def dispatch|def run_fullgraph|def run_pw_graph' vllm/v1/worker/gpu/cudagraph_utils.py
rg -n 'num_tokens_after_padding|query_start_loc_np\[num_reqs|CUDAGraphMode.FULL' vllm/v1/worker/gpu/model_runner.py
```

## 9. 易错点与自我检验

1. `FULL_AND_PIECEWISE` 是每轮直接执行的单一模式吗？
2. 真实请求集合改变，是否必然重新 capture？
3. graph 命中是否只检查总 token 数？
4. padding token 应计入输出吞吐吗？
5. 7→6 个有效 token、仍 pad 到 8，能否直接预测加速 1/7？
6. 为什么 replay 前还有 offloader 的同步？

## 10. 本章总结

现在应能定位 MRV2 capture／dispatch／replay，区分实际与承载形状，解释合法回退和 padding 对实验的影响。尚未证明任何 graph 配置对本机有效，也未展开 torch.compile 编译器内部。

下一章把“候选数量改变计算形状”的问题落到投机解码：draft、verify、接受／拒绝、KV 和进度更新必须一起闭环，才能讨论预算优化。

### 自我检验答案

1. 不是，它是配置组合；单次运行用 NONE／PIECEWISE／FULL 之一。
2. 不必然，若元数据、缓冲区和描述符条件合法，可复用捕获结构。
3. 不是，还看请求和 query、uniform、LoRA、microbatch 等约束。
4. 不应；它不是对用户生成的有效 token。
5. 不能，各 kernel 对 padding 和变长 metadata 的响应不同，还存在 CPU 开销。
6. 为保证此前复制与图回放的缓冲区依赖顺序，capture 不能预先包含所有未来装载事件。
