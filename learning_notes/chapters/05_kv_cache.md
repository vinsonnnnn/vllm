# 第五章：从逻辑 token 到物理 KV，再到前缀复用

固定源码 commit 见首页。本章先沿单组 full attention、单 GPU、无投机的主路径讲分配，再打开 prefix cache；混合模型的多组协调只解释必要边界。

## 1. 本章要回答的问题

1. KVCacheManager 管的是 GPU 张量还是分配元数据？
2. Block Table 和 Slot Mapping 分别回答什么问题？
3. prefix cache 命中怎样影响待算 token 和块引用？
4. 请求结束后，KV 为什么可以“可回收但仍可命中”？
5. 当前版本的部分块和写时复制为什么不能忽略？

## 2. 先建立直观理解

把请求的 token 位置看作逻辑地址。一个请求的第 20 个位置，不等于显存数组中的第 20 个位置；多个请求可以分散占用物理 KV 块。

**块表（Block Table）**把“某请求的第几个逻辑块”映射到物理块号。**槽位映射（Slot Mapping）**进一步告诉本轮每个输入 token 的新 K、V 写到哪个物理位置。前者服务于历史 KV 的寻址，后者服务于本轮 KV 写入；二者相关但不是同一张表。

**引用计数**表示有多少活跃所有权占用某块；**前缀缓存索引**表示已知内容能否被其他请求匹配。释放活跃引用，不一定马上抹掉索引和显存数据。因此“free”通常指允许被重用，不等于立即归还显存给系统。

## 3. 与 nano-vLLM 的对应关系

可类比分页、逻辑块号和块复用。这里核心的资源元数据与 Worker 中的物理 KV 张量分离；协调器还负责多个 KV group。不能假定一个 block size 对所有层、后端、hash 粒度都相同，也不能把简化实现的整块缓存规则直接推广到当前仓库。

## 4. 源码阅读地图

| 顺序 | 路径与符号 | 为什么读 |
| --- | --- | --- |
| 1 | [vllm/v1/core/kv_cache_manager.py:228](../../vllm/v1/core/kv_cache_manager.py#L228)，`get_computed_blocks` | 查询可复用前缀，得到命中 token 数和块 |
| 2 | [vllm/v1/core/kv_cache_manager.py:343](../../vllm/v1/core/kv_cache_manager.py#L343)，`allocate_slots` | 容量检查、附加命中块、分配新块、登记缓存 |
| 3 | [vllm/v1/core/kv_cache_coordinator.py:66](../../vllm/v1/core/kv_cache_coordinator.py#L66)，`KVCacheCoordinator` | 看 manager 如何把工作分发给各组，不先展开全部混合模型 |
| 4 | [vllm/v1/core/block_pool.py:658](../../vllm/v1/core/block_pool.py#L658)，`get_new_blocks`；第 713、734 行 `touch`、`free_blocks` | 分配、命中占用与归还的计数规则 |
| 5 | [vllm/v1/core/kv_cache_utils.py:622](../../vllm/v1/core/kv_cache_utils.py#L622)，`hash_block_tokens` | hash 包含父前缀、当前 token 和额外区分信息 |
| 6 | [vllm/v1/core/single_type_kv_cache_manager.py:714](../../vllm/v1/core/single_type_kv_cache_manager.py#L714)，`FullAttentionManager.find_longest_cache_hit` | 先完整块匹配，再看有条件的细粒度匹配 |
| 7 | [vllm/v1/worker/gpu/block_table.py:17](../../vllm/v1/worker/gpu/block_table.py#L17)，`BlockTables`；第 191 行 `compute_slot_mappings` | 从核心块号进入 Worker 表，再为本轮 token 计算槽位 |
| 8 | [vllm/v1/worker/gpu/model_runner.py:1405](../../vllm/v1/worker/gpu/model_runner.py#L1405)，`prepare_attn` | 看块表与 slot mapping 的共同生产位置 |

## 5. 一次分配与使用的真实链路

1. **核心普通调用**：waiting 请求在 Scheduler 查询 `get_computed_blocks`；命中会减少该请求此次需要实际计算的前缀长度。
2. **核心普通调用**：`allocate_slots` 结合已计算／命中长度、新计算位置、lookahead 和资源保留检查容量，失败返回 None；成功才取得本轮所需块。它更新的是管理元数据，不在 CPU 中算 K、V。
3. **核心计划数据→Worker**：SchedulerOutput 携带新请求的完整块表或已有请求的块增量。uni 下本进程分发；使用其他执行器时边界不同。
4. **Worker 普通调用→GPU 输入准备**：`BlockTables.append_block_ids`、`apply_staged_writes` 更新持久表；`gather_block_tables` 按本轮请求顺序聚合；`compute_slot_mappings` 发起 Triton 工作，使用本轮 positions 得到物理槽位。
5. **CPU 发起 GPU attention 工作**：forward context 将 metadata 和按层 slot mapping 交给注意力层，后端完成 KV 更新与读取。缓存索引命中本身不会执行模型。
6. **核心收尾**：正常结束减少引用；异步仍有 GPU 写入时可延迟归还，不能让别的请求重用仍在写的物理块。命中索引是否保留与是否被驱逐是另一问题。

## 6. 关键源码精读

### 6.1 不能让全部输入都省掉计算

`KVCacheManager.get_computed_blocks`：

<!-- source: vllm/v1/core/kv_cache_manager.py:258-263 -->

```python
max_cache_hit_length = request.num_tokens - 1
computed_blocks, num_new_computed_tokens, num_uncached = (
    self.coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
)
```

输入为请求 token 对应的 hash；输出为可复用块、命中长度及协调所需信息。把命中上限留在最后一个 token 之前，是为了重新得到生成所需 logits。实际重算可能受块／hash 对齐扩大，不能一概说“只重算最后一个 token”。

### 6.2 命中是增加所有权

`BlockPool.touch` 的关键部分：

<!-- source: vllm/v1/core/block_pool.py:721-726 -->

```python
for block in blocks:
    # ref_cnt=0 means this block is in the free list (i.e. eviction
    # candidate), so remove it.
    if block.ref_cnt == 0 and not block.is_null:
        self.free_block_queue.remove(block)
    block.ref_cnt += 1
```

输入是命中的块；无新模型输出。它把即将被活跃请求使用的块从可驱逐队列移走并增加引用；后面的 attention 读取这些 KV。**设计解释：**先查到 hash 不等于已经拥有安全读取权；分配路径必须把命中和引用生命周期接上。

### 6.3 token 的物理槽位

`_compute_slot_mappings_kernel`：

<!-- source: vllm/v1/worker/gpu/block_table.py:339-348 -->

```python
block_indices = tl.where(
    mapping_enabled, local_positions // kernel_block_size, 0
)
block_offsets = local_positions % kernel_block_size
block_numbers = tl.load(
    block_table_ptr + req_state_idx * block_table_stride + block_indices,
    mask=is_local,
    other=0,
)
slot_ids = block_numbers * kernel_block_size + block_offsets
```

输入是请求持久行、本地 token 位置和后端块粒度；输出 slot_ids 之后写入映射张量。当前主线 CP_SIZE=1，无需展开 context parallel 的位置变换。padding 或禁用写映射的情况在后续代码使用 PAD_ID，不能把填充位置当成有效请求写入。

### 当前版本的额外事实：部分块与 CoW

**写时复制（copy-on-write，CoW）**指共享内容需要继续写入时，先转到私有副本，避免破坏其他读者的缓存。当前 FullAttentionManager 的第 745 行起明确区分细粒度模式：当 alignment 比物理块小且满足整除关系时，可探测第一个未完整命中块的内部 hash 边界。

其 `_cache_partial_tail_block` 入口见 [vllm/v1/core/single_type_kv_cache_manager.py:823](../../vllm/v1/core/single_type_kv_cache_manager.py#L823)，私有块重定向见 [vllm/v1/core/single_type_kv_cache_manager.py:425](../../vllm/v1/core/single_type_kv_cache_manager.py#L425) 的 `_apply_cow`。它保留复制两端引用并记录待执行复制；后续 Worker 完成复制前不能回收端点。这里不把细粒度模式当作所有模型默认开启，也不在本章展开 Mamba 状态迁移。

所以应区分三件事：已知 token 可算 hash、已登记可复用缓存、GPU 相应写入已完成。源码允许调度阶段先登记元数据，安全性还依赖执行顺序和在飞行状态；不能把 hash 表当成独立于调度的即时完成证明。

## 7. 贯穿案例：同一前缀再次到来

**教学示例，非日志。**先在第一章的块 7、block size 16 上扩展 prompt 为 18 个 token；本节使用整块对齐的单组情形，开启 prefix cache，无细粒度部分命中。

| 请求／步骤 | 逻辑块→物理块 | 计算与 KV |
| --- | --- | --- |
| R 首次 prefill | 逻辑块 0→7，块 1→9 | 计算 18 个 prompt token；块 7 的 16 个位置完整，块 9 只有部分有效 |
| R 完成后 | 活跃引用释放 | 块 7 的 hash 与数据可能仍保留，可被驱逐，不保证永远留存 |
| S 有相同前 16 个 token、总 prompt 长 18 | 复用物理 7；尾部使用新可写块 12 | 假设最长命中 16，实际计算 positions 16、17 |
| S 第一个输出 55 | 同上 | 55 尚未作为输入，下轮位置 18 才需要 KV |

S 的 position 16：逻辑块索引 `16//16=1`，块内偏移 0；查 S 块表得到物理 12，因此 slot=`12*16+0=192`。下一 position 17 的 slot=193。这里的 slot 是教学单组展平索引，不是跨所有层通用的字节地址。

若 prompt 恰长 16，命中上限是 15；在本节整块对齐设置下，不能复用整段 16 来完全跳过 forward。换成细粒度模式后须重新按照 alignment 算命中，不能沿用这个整块数值结论。

## 8. 动手实验

**已验证**：源码和摘录；教学 slot 数值为手算。**仅提供步骤**：环境具备测试依赖时运行现有 manager suite：

```bash
uv run --no-project --offline --python .venv/bin/python python -m pytest tests/v1/core/test_single_type_kv_cache_manager.py -q
uv run --no-project --offline --python .venv/bin/python python -m pytest tests/v1/worker/test_gpu_block_table.py -q
```

前者的具体用例和 fixture 仍需检查，后者涉及 GPU 张量；两者本轮均未运行。GPU 可用后，可用相同 prompt 连续两次提交，在禁用与启用 prefix cache 的独立实例中对照输出和缓存命中统计；先保证输出契约，再分析 prefill 工作量，不能只凭第二次 wall time 较短归因 prefix cache。

无 GPU 替代练习：写出块表 `[7,12]` 在 positions 14、15、16、17 的槽位（126、127、192、193），再说明为何本轮只应提交实际新计算位置的 slot。用以下命令定位引用计数和部分块路径：

```powershell
rg -n 'def touch|def free_blocks|is_block_writable' vllm/v1/core/block_pool.py
rg -n 'fine_grained|_cache_partial_tail_block|_pending_cow_copies' vllm/v1/core/single_type_kv_cache_manager.py
```

## 9. 易错点与自我检验

1. Block Table 与 Slot Mapping 是否可以互换传给后端？
2. ref_cnt=0 为什么还能命中缓存？
3. 同样的块内 token，前面的上下文不同，为什么不能简单复用？
4. prefix 全命中为什么仍要执行计算？
5. 分配了一块是否说明它每个位置的 KV 都有效？
6. 当前版本是否永远只缓存完整物理块？
7. 结束请求为什么不一定立即调用 GPU allocator 释放一段显存？

## 10. 本章总结

现在应能沿 Scheduler→KV manager→pool→Worker 块表→slot mapping→attention 解释资源如何变成地址，区分引用、缓存索引和设备有效内容。细粒度、混合组与 CoW 已有入口，尚未系统覆盖其全部算法与分布式 KV。

下一章用这些地址解释 Runner 如何把多请求组织成实际模型输入，以及后端为什么还需要 query_start_loc、seq_lens 和 attention metadata。

### 自我检验答案

1. 不能；一张按请求／逻辑块映射历史，一张按本轮 token 映射写入位置，形状和消费方不同。
2. 没有活跃所有者但尚未被驱逐；命中后 touch 重新增加引用。
3. hash 链含父前缀，注意力 K/V 内容依赖完整因果上下文，还需考虑额外区分信息。
4. 要得到下一 token 的 logits；重算范围受 alignment 影响。
5. 不说明，分配容量与实际计算进度分开。
6. 不是；存在有条件的细粒度部分命中与部分 prompt 尾缓存，要核查配置和 CoW。
7. 常驻 KV 张量按块池复用；释放通常减少逻辑占用，且在飞行写入还可能要求延迟归还。
