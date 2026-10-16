# 第六章：Model Runner 如何把计划变成模型输入

固定版本沿用首页。本章主角是 `vllm/v1/worker/gpu/model_runner.py` 中的 MRV2；MRV1 只作为分支路标，不将其 InputBatch 实现混进来。

## 1. 本章要回答的问题

1. 一轮 SchedulerOutput 如何改变 Worker 的持久状态？
2. 本轮 batch 行号与请求的持久行号为什么不同？
3. input_ids、positions、query_start_loc、seq_lens 分别如何产生？
4. Attention Backend 从哪里得到 KV、块表和 slot mapping？
5. forward 与 sample 之间保存了什么，为什么不能调乱顺序？

## 2. 先建立直观理解

**持久状态**跨多个 step 保留，例如请求的 token、块表和最近采样结果；**本轮输入**只描述当前选择的请求和位置。两者分开后，Runner 可以保留请求的固定槽位，再按本轮顺序汇聚输入，不用每轮从头构造全部请求信息。

**Attention Backend（注意力后端）**是注意力计算的具体实现及其 metadata 协议。模型的 Q、K、V 是计算内容；块表、序列边界和槽位是解释这些内容所需的地址与布局信息。仅有一大串 token ID，后端无法知道谁能注意到谁。

## 3. 与 nano-vLLM 的对应关系

可以类比 Model Runner 的 prepare inputs→forward→sample。这里需要额外理解“永久请求行→本轮 batch 行→展平 token 行”三个索引空间，以及 MRV2 的模型状态、采样状态、块表等协作对象。不要借用 MRV1 的 `CachedRequestState` 解释 MRV2 的持久行机制。

## 4. 源码阅读地图

| 顺序 | 路径与符号 | 为什么读 |
| --- | --- | --- |
| 1 | [vllm/v1/worker/gpu/model_runner.py:1055](../../vllm/v1/worker/gpu/model_runner.py#L1055)，`add_requests`；第 1113 行 `update_requests` | 新请求初始化和已有请求增量有不同输入 |
| 2 | [vllm/v1/worker/gpu/states.py:9](../../vllm/v1/worker/gpu/states.py#L9)，`RequestState` | 核对 GPU Runner 的持久行，不是前端同名类 |
| 3 | [vllm/v1/worker/gpu/model_runner.py:1149](../../vllm/v1/worker/gpu/model_runner.py#L1149)，`gather_batch_req_state` | 找请求排序与 idx_mapping |
| 4 | [vllm/v1/worker/gpu/model_runner.py:1202](../../vllm/v1/worker/gpu/model_runner.py#L1202)，`prepare_inputs` | 输入总长、分段、位置、采样索引与 padding |
| 5 | [vllm/v1/worker/gpu/input_batch.py:42](../../vllm/v1/worker/gpu/input_batch.py#L42)，`InputBatch`；第 327 行 `_prepare_pos_seq_lens_kernel` | 用张量字段反推每个索引含义 |
| 6 | [vllm/v1/worker/gpu/model_runner.py:1405](../../vllm/v1/worker/gpu/model_runner.py#L1405)，`prepare_attn`；[vllm/v1/worker/gpu/attn_utils.py:259](../../vllm/v1/worker/gpu/attn_utils.py#L259)，`build_attn_metadata` | 接回第五章的物理映射 |
| 7 | [vllm/v1/attention/selector.py:102](../../vllm/v1/attention/selector.py#L102)，`get_attn_backend` | 后端选择受模型、dtype、KV 与平台等条件影响 |
| 8 | [vllm/model_executor/models/opt.py:114](../../vllm/model_executor/models/opt.py#L114)，`OPTAttention.forward`；[vllm/model_executor/layers/attention/attention.py:483](../../vllm/model_executor/layers/attention/attention.py#L483)，`Attention.forward` | 从模型层走到通用 attention 包装 |
| 9 | [vllm/model_executor/layers/attention/attention.py:750](../../vllm/model_executor/layers/attention/attention.py#L750)，`unified_attention_with_output` | 拿上下文调用选中的 impl.forward |
| 10 | [vllm/v1/worker/gpu/model_runner.py:1896](../../vllm/v1/worker/gpu/model_runner.py#L1896)，`sample_tokens` | 消费 ExecuteModelState，再推进采样与状态 |

## 5. 真实调用链

`Worker.execute_model` **普通调用** MRV2 `execute_model`。其准备阶段先消费清理信息、加入／更新请求，再汇聚本轮状态，选择执行描述，构造 InputBatch 和 attention metadata。若没有实际 token，也存在清理／connector 路径，不能无条件执行 max／forward。

输入准备中既有 CPU 数组组织，也有 **CPU 发起的 GPU kernel**：例如 positions／seq_lens、slot mapping 都可以由 Triton 产生。然后 `set_forward_context` 建立当前层能读取的上下文；模型 eager 普通调用触发底层设备计算。

对 OPT 这个具体模型：`OPTAttention.forward` 做 qkv_proj，切分 q、k、v，调用 `self.attn`，再 out_proj。通用 Attention 根据后端决定 KV update 是否单独调用，最后通过 custom op 或 direct call 进入 `impl.forward`。到此才应该依据实际选中后端继续往内核追，而不是跳到任意名为 paged_attention 的 C++ 文件。

forward 之后保存 ExecuteModelState。核心随后通过执行器调用 `sample_tokens`：读取并清掉这个中间状态，计算 logits、采样、推进 Worker 的 token 状态，构造可回收输出。第七章再解释复制流与回收事件；此处没有新增 ZMQ 边界，前端／核心通信仍沿第一章。

## 6. 关键源码精读

### 6.1 本轮顺序不等于持久行顺序

`gather_batch_req_state`：

<!-- source: vllm/v1/worker/gpu/model_runner.py:1166-1176 -->

```python
draft_tokens = scheduler_output.scheduled_spec_decode_tokens
# batch_idx -> req_id
req_ids = sort_batch_req_ids(
    num_tokens_per_req, draft_tokens, self.decode_query_len
)

numtoks_iter = map(num_tokens_per_req.__getitem__, req_ids)
num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

idx_mapping_iter = map(self.req_states.req_id_to_index.__getitem__, req_ids)
idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.intp, count=num_reqs)
```

输入是每请求工作量和候选信息，输出 req_ids、各请求长度与 batch→持久行映射。后面的张量汇聚必须共用这个映射，不能对一张表排序、另一张仍按旧顺序。**设计解释：**保持持久槽位的稳定，同时允许本轮按执行需要排序，可减少无关状态搬动。

### 6.2 位置与总上下文长度

在 `_prepare_pos_seq_lens_kernel` 中，已经根据 idx_mapping 取出该请求此前的 computed，并取出当前 query 分段：

<!-- source: vllm/v1/worker/gpu/input_batch.py:353-360 -->

```python
seq_len = num_computed_tokens + query_len
tl.store(seq_lens_ptr + req_id, seq_len)

for i in tl.range(0, query_len, BLOCK_SIZE):
    block = i + tl.arange(0, BLOCK_SIZE)
    mask = block < query_len
    pos = num_computed_tokens + block
    tl.store(pos_ptr + start + block, pos, mask=mask)
```

输入是 Worker 为本轮使用的旧进度和 query_len；输出 seq_lens 和逐 token positions。不要误用 Scheduler 已预记账后的新 computed 去再加一次 query_len；SchedulerOutput 对 Worker 传递的是准备本轮输入所需的进度，第三章的两侧状态不可混成一个变量。

### 6.3 KV 更新是否在 attention forward 内，要看后端

Attention 的 direct-call 分支：

<!-- source: vllm/model_executor/layers/attention/attention.py:538-546 -->

```python
if (
    not self.attn_backend.forward_includes_kv_cache_update
    and self.kv_sharing_target_layer_name is None
    and key is not None
    and value is not None
):
    kv_cache_dummy_dep = unified_kv_cache_update(
        key, value, self.layer_name
    )
```

输入是新 K/V 及当前层；必要时先调用 KV 写入实现，并创建依赖标记，供后续 attention 保持顺序。该函数会从 forward context 取得实际 kv_cache 和 layer_slot_mapping，入口见 [vllm/model_executor/layers/attention/attention.py:706](../../vllm/model_executor/layers/attention/attention.py#L706)。custom-op 分支有相应行为。不能声称所有后端都在一个相同 kernel 内写 KV 并做 attention。

## 7. 两请求输入组织案例

**教学示例。**沿第四章第二轮：A 的下一输入是 55，B 要计算前三个 prompt token。假设本轮顺序是 `[A,B]`，持久行分别为 2、0；索引数字仅供演示，不承诺真实排序结果。

| 张量／数据 | 教学值 | 含义 |
| --- | --- | --- |
| idx_mapping | `[2,0]` | 本轮第 0、1 个请求分别取持久行 2、0 |
| num_scheduled_tokens | `[1,3]` | A 一位置，B 三位置 |
| query_start_loc | `[0,1,4]` | 展平输入中 A 为 `[0,1)`，B 为 `[1,4)` |
| input_ids | `[55,b0,b1,b2]` | b0 等是符号化的教学 token |
| positions | `[4,0,1,2]` | 各请求独立的逻辑位置 |
| seq_lens | `[5,3]` | 各请求这轮可见的总长度，绝非各自 query 长度 |
| 有效输入数 | 4 | graph padding 之后的张量行数可能更大 |

这组张量和对应块表一起构成 attention 的寻址／分段契约。B 的 prompt 尚未完成，因此即使通用计算产生了中间预测行，也不应提交正式输出；有效采样数的处理入口见 [vllm/v1/worker/gpu/input_batch.py:491](../../vllm/v1/worker/gpu/input_batch.py#L491) 的 `_get_num_sampled_and_rejected_kernel`。

A 的新 K/V 写入第五章计算的 position 4 对应槽位；B 的写入按自己的块表和 positions 0—2 计算。两请求共享展平张量不意味着共享因果上下文。

## 8. 动手实验

**已验证**：本章静态导航、摘录和行号。**仅提供步骤**：运行既有 MRV2 输入准备测试：

```bash
uv run --no-project --offline --python .venv/bin/python python -m pytest tests/v1/worker/test_gpu_input_batch_v2.py -q
```

前置条件是完整 GPU 测试依赖与支持的 CUDA 环境；本轮未运行。GPU 推理就绪后，先用第一章程序确认实际 MRV2，再从日志／选中 backend 配置确定注意力实现，然后只继续追这个 impl 的 forward。不要因为显卡型号就预设一定选 FlashAttention 的某个版本。

无 GPU 的练习：把本轮顺序换成 `[B,A]`，重新写出 idx_mapping、query_start_loc、positions、seq_lens 与按 batch 顺序汇聚的块表；原持久行号不变。用下列检索核对：

```powershell
rg -n 'sort_batch_req_ids|idx_mapping|prepare_pos_seq_lens|prepare_attn' vllm/v1/worker/gpu/model_runner.py
rg -n 'forward_includes_kv_cache_update|impl.forward' vllm/model_executor/layers/attention/attention.py
```

预期你能说明所有随 batch 重排的数据，而不是只重排 input_ids。

## 9. 易错点与自我检验

1. 三种行号分别是什么？
2. B query 长 3，为什么将来 seq_len 可能为 103？
3. 为什么不能把 Scheduler 预记账后的进度直接拿来当本轮起始 position？
4. Slot Mapping 是否足以代替所有 attention metadata？
5. 模型 forward 返回 hidden states 后，为什么不能直接返回用户字符串？
6. 从 OPTAttention 应如何继续定位实际 backend？

## 10. 本章总结

现在应能解释持久状态如何汇聚为一轮输入、各索引空间如何对齐、模型层如何使用 attention context，以及 forward 和 sample 的交接。未展开具体 backend kernel 的数学／布局优化，未断言本机实际选择了某后端。

下一章让 CPU 不必每轮等所有输出：复用本章的 Worker 持久状态和设备输入，解释异步调度、复制事件、批次队列和安全回收。

### 自我检验答案

1. 请求持久行、本轮 batch 行、展平 token 行；需要 idx_mapping 与 query_start_loc 关联。
2. 若已有 100 个上下文 token，本轮再算 3 个，总可见长度是 103。
3. 会把本轮位置重复累加，导致错位读写 KV；应使用协议提供的本轮旧进度。
4. 不足，还需请求边界、上下文长度、backend 特定 metadata 等。
5. 还要 compute_logits、采样、结果回收、停止处理与反分词。
6. 沿 self.attn→通用 Attention→后端选择／上下文→impl.forward；运行后确认具体实现，再继续 kernel。
