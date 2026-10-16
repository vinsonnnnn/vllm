# 增量术语表

本表只收录已出现的概念。首次解释以章节为准；后续章节补充精确字段与状态规则，不一次性展开所有后端。

| 术语 | 中文含义与本版作用 | 首见 |
| --- | --- | --- |
| Entrypoint | 用户入口，例如离线 Python `LLM`；将外部调用接入引擎 | 01 |
| V1 | 本仓库实际使用的引擎架构；公开 LLMEngine 是 V1 类别名 | 01 |
| LLMEngine | 前端引擎对象，组织请求输入与输出，通过 EngineCoreClient 接入核心 | 01 |
| EngineCore | 引擎核心，持有 Scheduler 和执行器，组织调度、执行和状态更新 | 01 |
| EngineCoreProc | EngineCore 的进程版本，增加通信线程、队列与忙循环 | 01 |
| IPC / ZMQ | 进程间通信／ZeroMQ 消息机制；SyncMPClient 与核心进程通过 socket 交换消息 | 01 |
| Queue | 队列；本章图中的 input/output queue 在线程之间交接对象，不自动表示跨进程共享内存 | 01 |
| Scheduler | 调度器；决定本轮哪些请求计算多少 token，并消费模型结果更新状态 | 01 |
| Token budget | token 预算；限制一轮可安排的工作量，不是“返回给用户的 token 数” | 01 |
| Executor | 执行器；统一调度侧调用 Worker 的接口，具体实现决定本进程调用或其他执行方式 | 01 |
| UniProcExecutor | 单进程执行器；Worker 留在核心所在进程，不额外推出“每 GPU 一个新进程”的结论 | 01 |
| Worker | 工作对象；管理设备、模型加载和执行入口，委托 Runner 准备并运行计算 | 01 |
| Model Runner / MRV2 | 模型执行器／其第二套实现；位于 V1 架构内部，不能与 V0/V1 引擎版本混淆 | 01 |
| Prefill / Decode | 提示词计算／逐步生成；用于解释工作负载，本版 Scheduler 不以两个独立 phase 切换作为统一抽象 | 01 |
| KV Cache / Block | 注意力 K、V 缓存／其分配块；分配一个块不表示块中每个位置都有有效 KV | 01 |
| num_computed_tokens | 已计入计算进度的 token 数；本章仅在关闭异步、无投机的教学案例中推演，其完整更新时序待后续章节核实 | 01 |
| Eager | 即时执行；本章用 `enforce_eager=True` 简化图执行路径，CUDA 内核仍在 GPU 上执行 | 01 |
| CUDA Graph | CUDA 图；记录并重放执行以减少重复发起开销，第八章讨论 capture、形状与 padding | 01 |
| Sampling | 采样；从 logits 选择输出 token，前向返回 hidden states 不等于已返回用户 token | 01 |
| Detokenization | 反分词；把 token ID 转为文本，是前端输出链的一部分 | 01 |
| TTFT / Throughput | 首 token 延迟／吞吐；第十章分别定义测量窗口和负载条件 | 01 |

## 第二至十章新增与精化

| 术语 | 中文含义与当前源码中的作用 | 首次深入 |
| --- | --- | --- |
| Renderer | 输入渲染器；纯文本路径在此正规化输入、tokenize 并构造 EngineInput | 02 |
| EngineCoreRequest | 发往核心的请求消息；与核心可变 Request 通过字段关联，不能依赖对象身份 | 02 |
| External / internal request ID | 外部／内部请求标识；前者面向调用者，后者用于引擎路由与唯一性 | 02 |
| FINAL_ONLY | 只对外产出最终结果；中间 token 仍被内部输出状态消费 | 02 |
| Stop token / stop string | 终止 token／字符串；分别由核心 token 检查和前端增量文本检查处理 | 02 |
| RequestState | 同名但不同类：前端保存输出状态；MRV2 `gpu/states.py` 保存持久请求行，不可混用 | 03 |
| SchedulerOutput | 一轮计划；包含每请求安排量、新请求信息、增量与清理 ID | 03 |
| ModelRunnerOutput | 一轮回收结果；包含 ID 顺序映射与各请求实际有效 token | 03 |
| num_in_flight_tokens | 已安排而输出尚未处理的位置数；与乐观 computed 共同描述进度 | 03 |
| Preemption | 抢占；回收／安排回收 KV、回队列并重算，保留已提交 token，不等于请求取消 | 03—04 |
| Chunked prefill | 分块 prompt 计算；部分 prompt 可与其他请求 decode 同轮安排，未完成的 chunk 不提交正式输出 | 04 |
| input_budget / token_budget | 本轮输入／调度额度；在 Scheduler 内由不同配置初始化，投机 drafting 槽位可影响前者扣减 | 04 |
| Hash chain | 前缀 hash 链；当前块 hash 包含前缀链信息，不能只按局部 token 匹配上下文 | 05 |
| CoW | 写时复制；部分缓存共享后需继续写时，转为私有块并保护复制两端生命周期 | 05 |
| KV group | KV 缓存组；协调器按组管理不同存储约束；单组教学例子不能直接外推到混合模型 | 05 |
| idx_mapping | 本轮 batch 行到 MRV2 持久请求行的映射，允许执行重排而不搬动全部持久状态 | 06 |
| query_start_loc | 展平 query 张量中各请求的累计分段边界，长度通常为请求数加一 | 06 |
| positions / seq_lens | 各输入 token 的逻辑位置／该请求本轮总可见长度；与 query 长度不同 | 06 |
| Forward context | 前向上下文；将按层 metadata、KV 和 slot mapping 接到 attention 实现 | 06 |
| Placeholder | 占位位置；CPU 用于异步预记账，Worker 中的实际 token 另行保存，不将 -1 当正常 token 输入 | 07 |
| D2H / copy event | 设备到主机复制／复制完成事件；CPU 转列表前须满足相应完成条件 | 07 |
| Stale output | 旧状态的晚到输出；抢占／取消后要按协议处理，不能任意扣减新状态计数 | 07 |
| Capture / replay | 捕获／回放；记录并复用设备执行结构，仍受缓冲区形状和流依赖约束 | 08 |
| BatchExecutionDescriptor | MRV2 本轮图模式与承载形状描述；匹配维度不只 token 数 | 08 |
| Draft / verify | 草稿提议／目标验证；前者产生候选，后者决定可提交前缀 | 09 |
| Correction / bonus | 纠正／额外 token；首次拒绝时由 target 纠正，全部接受时还可得到一个额外预测 | 09 |
| Dynamic SD | 此版按 batch-size 区间配置查表选择 K 的既有路径 | 09 |
| Adaptive Verification | 此版结合置信度／成本进行验证预算与压紧的既有路径，配置当前限制 DSpark | 09 |
| TPOT / ITL | 首 token 后平均每输出 token 时间／流式事件间隔；投机多 token 提交时尤须区分 | 10 |
| Goodput | 满足预先约定服务条件的有效完成速率；条件、窗口和失败统计都要固定 | 10 |
| Warmup / measurement window | 预热／测量窗口；初始化、capture 与稳态生成应分别记录 | 10 |
| Oracle / ablation | 正确性对照／消融；分别确认行为符合约束以及哪一部分产生收益 | 10 |
