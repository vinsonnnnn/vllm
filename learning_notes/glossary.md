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
