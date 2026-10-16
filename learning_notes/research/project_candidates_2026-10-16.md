# vLLM 非算子优化项目选题：18 个论文与开源方法候选

核实日期：2026-10-16。面向从 nano-vLLM 过渡到 vLLM、希望完成推理系统实习项目的读者。目标是复现公开方法，完成系统集成、正确性验证和性能实验；算法贡献归原作者，工程贡献单独说明。

## 1. 版本、环境与证据边界

| 项目 | 本文采用的事实 |
| --- | --- |
| 学习源码基线 | `f9c98bd1ebbddb9acedb1e6d7505ad9588f9a266`，不升级依赖、不切换版本 |
| 本轮开始时本地 HEAD | `5b3ee5181ebecbdbe95d91364077a5651149f59a`，是上述基线的文档提交后代 |
| 代码路径 | V1；重点关注单 GPU 文本生成，MRV2 与回退路径分别核实 |
| 已记录硬件 | RTX 4090 Laptop，16376 MiB 显存；不能按桌面 4090 24GB 估算容量 |
| 运行能力 | 现有环境尚缺 vLLM、PyTorch 等运行依赖；本轮只做资料和源码检查，没有运行任何候选算法 |
| 远端核实 | 查询上游 issue/PR，读取作者仓库和论文；查询及仓库 commit 记录见配套 JSON |
| 交付范围 | 选题材料；未安装作者依赖、下载模型、复制作者实现、创建上游 issue 或 PR |

环境详情沿用[笔记首页](../README.md)。未来开展贡献时，应另建工作目录，选定当时的上游 commit 和依赖快照；学习基线保持固定。本文给出的硬件可行性和接入方案都是**工程判断，尚待实测**。

证据按以下方式理解：

- **本地未检出**：在固定仓库的相关源码、测试和文档中，未找到方法名称或明确入口；不能证明不存在等价实现。
- **同名 PR 未检出**：上游方法名搜索没有命中，不能排除改名实现、未公开分支或搜索漏检。
- **已有外部实现**：作者 fork、独立引擎或中间件已经实现，贡献应定位为移植、完善兼容或协作，不能声称首次实现。
- **已有相近开放 PR**：暂停独立同功能 PR，先辨明差异并与已有工作协调。
- **作者代码未确认**：论文可读，但不能承诺能直接移植作者代码。

所有“最小项目”“PR 切口”和优先级均为本文的建议，不是维护者承诺。没有采用论文中的加速数字作为本机预期值，也没有根据论文被接收推断代码成熟度。

## 2. 怎样选择：先看贡献边界

“不做算子优化”在本文中指不开发或调优 CUDA、Triton、HIP、CuteDSL 算子。可以调用已经存在、许可允许且硬件兼容的算子，但复用它们仍需要构建和兼容验证。若移植过程中必须修改底层算子，便需要缩小项目或换方向。

| 级别 | 含义 | 适合的开始方式 |
| --- | --- | --- |
| A | 优先开展可行性验证，有明确系统工程边界 | 先完成一个最小验证，再决定是否做完整集成 |
| B | 值得做，但算子依赖、模型资源或相近工作需先解决 | 先审计依赖和接口，不直接承诺性能 |
| C | 协作、生态组件或较高风险研究储备 | 先复现或讨论；不重复创建同功能模块 |

### 18 个候选总览

年份区分首发和更新，不把 2024 年预印本包装成 2026 年新算法。

| 编号 | 方法与时间 | 主要模块 | 作者代码情况 | 非算子边界及贡献位置 | 建议 |
| --- | --- | --- | --- | --- | --- |
| P01 | SINQ，2025；ICML 2026 | 权重量化加载与执行 | 有，vLLM 集成仍列为进行中 | 检查点、布局、缩放；复用后端是否成立待验证 | A |
| P02 | SLAI，2025 | 请求调度与延迟目标 | 有，基于 Sarathi-Serve | decode 紧迫度与 prefill 分配；区别已有 SLO PR | A |
| P03 | Fast KVzip，2026-01 | KV 重要性预测与压缩 | 有，另有 KVpress 参考实现 | 评分和缓存生命周期；物理压缩后端是前置门槛 | B |
| P04 | KVzap，2026-01 | 自适应 KV 剪枝 | 有，NVIDIA KVpress | 预测器集成与阈值策略；逐 head 布局适配待验证 | B |
| P05 | SAM Decoding，2024；ACL 2025 | 检索式 proposer | 有，独立实现 | CPU 检索和请求状态；须比较 suffix 与现有 GPU PR | B |
| P06 | NanoSpec，仓库标注 ICML 2026 | 动态草稿词表 | 有，作者引擎含定制算子 | 只考虑策略和状态适配；已有相近稀疏 LM Head PR | C |
| P07 | SSD / Saguaro，2026-03 | 异步 draft/verify | 有，独立引擎；社区有 vLLM 原型 | 消息、分支 KV、回退；优先多 GPU | C |
| P08 | JetSpec / JetSpec++，2026-06 / 10 | 树形并行投机 | JetSpec 有 vLLM fork；++ 发布完整性待确认 | 集成与执行组织；树形后端和分支状态复杂 | C |
| P09 | SpecFit，2026-10 | MoE 卸载与验证子树选择 | 有，实验性 | 专家传输成本与预算；在线路由成本证据不足 | C |
| P10 | SpecReason，2025 | 推理步骤级大小模型编排 | 有，PoC 使用旧 vLLM | 外部控制器优先；会改变中间推理与输出分布 | B |
| P11 | KVLink，2025 | 文档 KV 拼接与适配模型 | 有代码和小模型权重 | 位置、特殊 token、模型加载；需适配检查点 | B |
| P12 | EPIC，2024；ICML 2025 | 非前缀 KV 复用与重算 | 有，vLLM 0.7.0 功能 demo | 迁移缓存协议和生命周期；不可照搬旧引擎 | C |
| P13 | Continuum，2025；2026 更新 | 工具等待期间的 KV 保留 | 有，preview fork | 有直接相关保留 API PR，适合策略协作 | C |
| P14 | ThunderAgent，2026-02 | 程序感知调度与工具资源 | 有，已支持 vLLM/SGLang | 外部中间件已有，贡献兼容或新策略 | A，生态方向 |
| P15 | CONCUR，2026-01 | Agent 级准入控制 | 本轮未确认作者代码 | 可按论文复现控制层；先比较固定并发基线 | B |
| P16 | CacheScout，2026 | Agent 转移预测与预取 | 本轮未确认作者代码 | 学习缓存复用关系；先做 trace 回放 | C |
| P17 | SageSched，2026-03 | 不确定性感知调度 | 本轮未确认作者代码 | 长度分布预测和成本模型，非单纯预测一个长度 | C |
| P18 | EvoSpec，2026，v3 于 07 更新 | 动态词表与在线 LoRA | 本轮未确认作者代码 | 在线训练与 serving 隔离；已有相关词表工作 | C |

对于当前单卡条件，优先比较 **P01 SINQ、P02 SLAI、P14 ThunderAgent**；如果更想深入 KV 内存机制，选择 **P03 Fast KVzip** 的后端可行性验证。此排序衡量可开展的工程工作，不是论文先进性或 PR 接收概率。

## 3. 逐项项目卡片

### P01：SINQ——量化检查点接入

**依据。** [论文](https://arxiv.org/abs/2509.22944)研究利用 Sinkhorn 风格归一化和双向缩放进行免校准权重量化。[作者仓库](https://github.com/huawei-csl/SINQ)提供量化与加载流程、GemLite 执行路径，并仍将 vLLM/SGLang 集成列为进行中。本地未检出，同名上游 issue/PR 未检出；作者可能已有未公开工作。

**你可以做什么。** 限定 dense 文本模型、1D、4-bit、一个 group size，实现配置读取、权重与缩放加载、合并线性层适配和后端分派。首个验证对象是一个线性层，再扩到整模型。

**非算子门槛。** 用作者参考解量化验证矩阵结果，确认双向缩放可由已有后端和常规张量操作表达。若必须改变 GEMM 格式或写新内核，停止扩大范围。Pre-SINQ 加 AWQ/GPTQ 是另一条工具链路线，不等于新增原生 SINQ 执行模块。

**实验与交付。** 比较原模型、作者量化模型、vLLM 接入模型的数值与任务质量，再测显存、加载时间、prefill/decode 和并发性能。先用可容纳的小模型验证；不能把量化过程更快当成 serving 更快。可形成实质性量化支持 PR，但须先与作者协调。

### P02：SLAI——考虑逐 token 延迟约束的调度

**依据。** [论文](https://arxiv.org/abs/2508.01002)处理不同请求的延迟要求；[作者代码](https://github.com/agrimUT/SLAI)基于 Sarathi-Serve，包含实验脚本。它不是当前 vLLM 的直接插件。上游已有[按 TTFT 目标重排等待请求的 PR #53571](https://github.com/vllm-project/vllm/pull/53571)，因此不能再做一个同功能队列策略。

**你可以做什么。** 先复现论文的 decode 紧迫度判定，再将“哪些 decode 必须本轮运行、剩余容量怎样留给 prefill”适配到 V1。第一版使用两类静态延迟目标，关闭投机和复杂 connector，保留公平性约束。

**非算子门槛。** 只修改 CPU 决策与调度元数据；必须理解异步调度预记账，不能因暂缓某次 decode 错误回退请求状态。单卡小模型可以起步，但小模型实验不代表大模型同样受益。

**实验与交付。** 固定到达流与延迟目标，对照 FCFS、priority、适用的上游 SLO 策略；报告 TTFT、TBT 分布、目标达成率、goodput 和最长等待。若差异只是等待队列排序，应加入已有 PR，而不是独立重复实现。

### P03：Fast KVzip——轻量门控驱动的 KV 压缩

**依据。** [论文](https://arxiv.org/abs/2601.17668)通过训练门控模块估计 KV 重要性；[作者实现](https://github.com/Janghyun1230/FastKVzip)提供预训练 gate，包含 Qwen3-4B 对应权重。[KVpress 参考代码](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/fastkvzip_press.py)明确区分参考压缩与作者优化实现。本地未检出，同名 PR 未检出。

**你可以做什么。** 先复现 gate 分数与保留集合，再验证单请求、prefill 后压缩的最小路径。后续才考虑分块 prefill、decode 期间压缩和并发请求。

**非算子门槛。** 必须找到能复用的物理 gather/compaction 与注意力读取路径。逐 head 不同保留集合与 vLLM 分页布局不是天然兼容。只把分数低的 token 屏蔽，不代表物理块被释放；统一跨 head 选择则是方法变体，需另做质量对照。

**实验与交付。** 对照 FullKV 和作者方法，测评分、整理、注意力、总生成时间及实际块数。质量覆盖检索、推理和多轮信息保留。若后端可复用，可贡献压缩元数据与生命周期支持；否则先交付复现报告，不伪称完整集成。

### P04：KVzap——预测器驱动的在线 KV 剪枝

**依据。** [论文](https://arxiv.org/abs/2601.07891)使用快速预测模型近似重要性评分，覆盖 prefill 和 decode；[KVpress 实现](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/kvzap_press.py)建议结合 DMSPress 使用。本地未检出，同名 issue/PR 未检出。

**你可以做什么。** 先接预测器和阈值决策，与 KVpress 对照保留结果；随后处理达到预算时的物理回收、近期 token 保护及每个请求的预测器状态。

**非算子门槛。** 与 P03 共享分页布局难点，两个项目应先择一。预测器权重、模型架构和 head 配置要匹配；不能假设小模型直接使用 8B gate。原型可以复用张量操作，性能版是否需要新后端尚待核实。

**实验与交付。** 测剪枝频率、预测开销、实际 KV 字节和任务质量；验证批次重排、取消及重复分配后状态不串请求。PR 应围绕一种明确支持的模型/缓存组合，不把整套 KVpress 作为强制依赖塞入热路径。

### P05：SAM Decoding——后缀自动机作为草稿提供者

**依据。** [论文](https://arxiv.org/abs/2411.10666)与[作者代码](https://github.com/hyx1999/SAM-Decoding)使用后缀自动机从动态历史和静态语料检索候选。这是较成熟的 2024/2025 工作，不是最新月份算法。当前 vLLM 已有 ngram、suffix，且[PR #60339](https://github.com/vllm-project/vllm/pull/60339)正在增加 MRV2 的 GPU suffix 实现。

**你可以做什么。** 仅在 SAM 对特定代码编辑或重复文本负载有可测差异时，实现请求级动态状态、检索长度与候选 token 交接；先用线性候选和已有验证器，暂不引入树形验证。

**非算子门槛。** CPU 数据结构本身不需要新 GPU 算子，但 CPU/GPU 同步可能抵消收益。只用已经接受的 token 更新动态索引，正确处理拒绝、重算和取消；共享静态语料要有明确隔离策略。

**实验与交付。** 对照 AR、ngram、suffix 和适用的现有 PR；报告索引构建、CPU 时间、同步开销与端到端性能。若差异只是换索引、收益不足，保留为学习复现，不为新增算法名而提 PR。

### P06：NanoSpec——上下文感知的动态草稿词表

**依据。** [作者代码与论文引文](https://github.com/csAugust/NanoSpec)描述逐步构建较小活动词表、异步整理权重与 GPU 状态管理；仓库标注 ICML 2026，提供 EAGLE-2/3 入口。本轮没有确认独立论文页，引用以作者仓库为准。

**已有重叠。** [PR #59740](https://github.com/vllm-project/vllm/pull/59740)已实现上下文感知稀疏 LM Head；[PR #60387](https://github.com/vllm-project/vllm/pull/60387)处理通用缩小草稿词表。方法名搜索为零，不代表没有同方向工作。暂停独立“通用动态词表模块”PR。

**可研究差异。** 对比论文的活动词表更新规则与上述 PR 的静态列表加动态选行；先建立 token ID 映射、命中覆盖率和候选质量的参考实现。已有 EAGLE-3 检查点可能在训练时就使用固定子词表，必须核对检查点，不能假设总能进一步动态裁剪。

**非算子门槛与实验。** 作者引擎包含定制 CUDA 代码；只移植词表策略不保证保留性能。若现成 gather/GEMM 足够，测词表维护、整理、head 时间和完整轮次；否则进入原 PR 协作或保留算法对照，不开发新算子。

### P07：SSD / Saguaro——验证时提前准备下一轮候选

**依据。** [论文](https://arxiv.org/abs/2603.03251)与[作者引擎](https://github.com/tanishqkumar/ssd)并发起草与验证，预测可能的验证结果并缓存下一轮候选。作者异步示例使用独立 draft GPU。

**已有重叠。** 早期[issue #36037](https://github.com/vllm-project/vllm/issues/36037)因无活动关闭；更相关的是开放的[RFC #42109](https://github.com/vllm-project/vllm/issues/42109)，评论给出了[社区原型分支](https://github.com/laviier/vllm/tree/disagg_draft)。不能再把它描述成没有 vLLM 移植。

**你可以做什么。** 先审计社区原型的分支 KV 所有权、请求取消、超时与 cache miss 回退，从一个可复现缺口参与已有实现。一个 target GPU 加一个 draft GPU 的缩小实验，是建议起点，不是作者已验证的配置。

**实验与交付。** 对照相同 GPU 总预算的 AR、同步投机和异步投机，记录通信、候选缓存命中和浪费的起草工作。单卡只能做逻辑/模拟验证，不能替代独立硬件重叠实验。适合多 GPU 条件下的协作项目。

### P08：JetSpec / JetSpec++——并行树形起草与流水执行

**依据。** [JetSpec 论文](https://arxiv.org/abs/2606.18394)、[作者实现](https://github.com/hao-ai-lab/JetSpec)及[vLLM fork](https://github.com/JetSpec-project/vllm-jetspec)已经公开；[JetSpec++ 论文](https://arxiv.org/abs/2610.19792)于 2026-10-14 发布，增加自修正、前瞻起草与流水执行。现有 JetSpec 代码不自动证明 ++ 全部实现已发布。

**你可以做什么。** 对作者 fork 做功能差异审计，固定一个模型和贪心模式，验证树节点输入、mask、接受路径与 KV 回收，再评估适合上游的最小切分。

**非算子门槛。** 作者有树形注意力和图执行后端；若复用现成后端无法满足 vLLM 的接口，应暂停移植。把树改成单条链只能算消融版本，不能当成完整方法。

**实验与交付。** 对照作者 fork、线性并行起草和 AR，分解树构建、draft、verify、回收时间。首先确认 ++ 所需权重与代码发布完整性。当前更适合有较大显存与作者协作的后续项目。

### P09：SpecFit——按专家传输成本选择验证子树

**依据。** [论文](https://arxiv.org/abs/2610.15417)与[实验代码](https://github.com/Chlience/SpecFit)针对卸载 MoE，根据专家缓存状态和预计传输成本选择验证子树。代码公开了模型配置、结果和复现材料，同名上游 PR 未检出。

**关键限制。** 作者说明路由来自单独参考运行，路由获取不计入计时；预测准确度实验使用模拟预测。因此不能将其结果解释为在线预测器已经获得端到端收益。论文使用的桌面 4090 24GB 也不同于本机 16GB。

**你可以做什么。** 首先复现 CPU 子树选择与成本回放；随后验证在线可获得的路由信息，将预测、通信和决策开销纳入时钟。不要使用当前决策之后才知道的真实专家路由。

**实验与交付。** 对比固定树预算、只看接受收益、加入专家传输成本的策略。若在线路由条件不成立，报告失败条件而非包装成优化。当前是高风险研究候选，不建议直接承诺一个完整 vLLM PR。

### P10：SpecReason——推理步骤级大小模型协同

**依据。** [论文](https://arxiv.org/abs/2504.07891)与[作者 PoC](https://github.com/ruipeterpan/specreason)让小模型执行部分推理步骤、大模型评估和修正。PoC 使用 vLLM 0.8.2，并包含 V0 启动示例；需要迁移，不能照抄到固定 V1。

**你可以做什么。** 先实现 V1 兼容的外部控制器，明确步骤边界、接受阈值、超时和失败回退。可以在小模型组合上验证接口，但实际质量收益需独立评估。

**非算子门槛。** 编排本身无需新算子；多个模型的驻留、切换与 KV 复用是系统问题。语义级接受不同于 token 级拒绝采样，不能宣称保持目标模型原始输出分布。

**实验与交付。** 用任务正确率、每题耗时、生成 token 和失败回退率作 Pareto 对照，包含只用大模型及固定大小模型策略。更适合外部推理组件或作者项目贡献；只有发现通用引擎接口缺口，才考虑 vLLM 核心 PR。

### P11：KVLink——使用适配模型复用独立文档 KV

**依据。** [论文](https://arxiv.org/abs/2502.16002)与[作者代码](https://github.com/UCSB-NLP-Chang/KVLink)通过位置处理、可训练特殊 token 和适配训练缓解独立文档 KV 拼接后的质量问题，提供 1B/3B 模型链接。本地和同名 PR 未检出。

**你可以做什么。** 使用已发布适配检查点，先验证两个文档独立编码、位置调整和连接 token，再接入缓存存取。保留“全文正常 prefill”“直接拼接”“KVLink”三组质量对照。

**非算子门槛。** 若已有 attention/backend 能表达掩码和读取布局，可聚焦模型与元数据；否则不能保证无需后端变更。不能把适配模型的结果当作原始模型在任意拼接下无损复用。

**实验与交付。** 测文档预计算、命中后的 TTFT、跨文档问答准确率、文档顺序变化和多用户并发。缓存键需绑定模型 revision、模板和连接配置。适合模型适配与 KV connector 联合项目，先与现有 RAG 缓存实现划清范围。

### P12：EPIC——非前缀上下文缓存的旧原型迁移

**依据。** [ICML 2025 论文](https://openreview.net/forum?id=qjd3ZUiHRT)讨论位置无关缓存；[作者仓库](https://github.com/DerekHJH/epic)明确说明只公开功能 demo，可与 vLLM 0.7.0 比较，不能用 demo 代替论文完整性能系统。

**你可以做什么。** 先整理独立上下文缓存、拼接后恢复质量和局部重算的语义，再评估能否通过 V1 的 connector 和 runner 支持。第一版限定少量文档与一种注意力布局。

**非算子门槛。** 抽取策略与缓存协议，不搬旧 engine/scheduler。局部重算是否可用现成后端表达必须先验证；KV 的上下文依赖不会因调整 RoPE 位置而自动消失。

**实验与交付。** 对照全文重算、纯前缀缓存及适用的 CacheBlend，分别计入离线准备和命中路径开销。适合原型迁移研究，PR 可能更适合 LMCache 等生态项目，而非大改 vLLM 核心。

### P13：Continuum——工具等待期间的 KV TTL 与程序调度

**依据。** [论文](https://arxiv.org/abs/2511.02230)与[preview fork](https://github.com/Hanchenli/vllm-continuum)研究工具耗时、跨轮连续性和 KV 保留。上游[RFC #37003](https://github.com/vllm-project/vllm/issues/37003)、[PR #38514](https://github.com/vllm-project/vllm/pull/38514)已涉及范围优先级和 TTL，另有[扩展接口 RFC #59708](https://github.com/vllm-project/vllm/issues/59708)。

**你可以做什么。** 研究工具时长预测和保留策略，在现有 API 工作上协作。先用确定的工具延迟回放，之后再引入预测误差；不要另开同功能 TTL 保留 API PR。

**非算子门槛。** CPU 缓存策略与程序元数据可独立于算子，但必须限制保留总量和最长时间，并处理 session 结束、取消、服务重启与过期。不能将保留优先级未经验证地解释成绝不可驱逐。

**实验与交付。** 比较 LRU、固定 TTL、已知工具时长、预测工具时长；同时统计受保护会话和其他请求的尾延迟。命中率提升伴随排队恶化时，应报告完整权衡。适合已有 PR 的策略与验证贡献。

### P14：ThunderAgent——程序感知的调度和工具资源管理

**依据。** [论文](https://arxiv.org/abs/2602.13692)与[作者代码](https://github.com/ThunderAgent-org/ThunderAgent)将 program 作为调度单元，管理工具资源。它已作为中间件支持 vLLM/SGLang，因此无需再次“首次接入 vLLM”。

**你可以做什么。** 先运行单后端程序回放，再寻找一个有证据的缺口，例如客户端断开、工具超时、program 取消后资源回收或后端压力变化下的行为。选题具体缺口需要再次查作者仓库 issue/PR，本轮没有断言这些都是现存 bug。

**非算子门槛。** 外部服务、调度协议和生命周期管理无需算子开发。单卡小模型可开展兼容实验；多节点负载均衡结论需要多个实际后端，不能用单后端模拟冒充。

**实验与交付。** 使用可复现的工具等待与取消场景，测程序完成时间、队列、资源峰值和回收时延。贡献更适合 ThunderAgent 或 vLLM router 生态；这是完整系统工程项目，但不能在简历上写成 vLLM 核心新模块。

### P15：CONCUR——Agent 级准入的反馈控制

**依据。** [论文及算法说明](https://arxiv.org/html/2601.22705v1)用 KV 压力和命中反馈调节活跃 Agent 数量，避免长轨迹间反复驱逐。论文有版本间标题差异，统一按 arXiv ID 识别。本轮未确认作者代码。

**你可以做什么。** 依据论文复现外部控制器，显式维护 Agent 活跃、等待工具和完成状态，再适配 vLLM 可获取的指标。先确认信号的定义与聚合窗口；不要直接把其他引擎的树缓存指标套到 vLLM。

**非算子门槛。** 控制层无需算子开发，单卡可用小模型、长轨迹和受限 KV 容量验证机制。不能使用实际输出长度等未来信息决定当前准入。

**实验与交付。** 比较固定并发网格、简单阈值策略与论文控制器，报告完成量、job completion time、命中/重算和控制振荡。若只是固定并发调小，应如实说明。优先贡献外部控制层；核心接口变更需证明通用需求。

### P16：CacheScout——学习 Agent 转移关系来驱逐和预取

**依据。** [论文](https://arxiv.org/abs/2608.14624)以在线学习的 Agent 执行关系预测未来上下文复用，并声称实现于 vLLM 之上。本轮未确认作者代码，也未确定其所有前提在固定版本成立；同名上游 PR 未检出。

**你可以做什么。** 先做按时间顺序的 trace 回放，区分上下文身份、每次调用和跨轮 session。对比 LRU、频率策略、带成本的策略，再考虑真实 KV 预取。

**非算子门槛。** 预测与驱逐策略无需新算子；真实预取必须复用 connector 的传输及完成通知，不能把提前读磁盘视为 KV 已在 GPU 可用。共享前缀引用和多租户边界需要设计。

**实验与交付。** 在独立测试轨迹上评估字节/token 加权命中、重算、无用预取、传输带宽和端到端延迟。只有真实 vLLM 推理才能证明性能；论文代码未确认前，定位为阅读和复现候选。

### P17：SageSched——用需求分布而非单点长度做调度

**依据。** [论文](https://arxiv.org/abs/2603.07917)结合 prompt、历史结果和计算/内存成本建模，考虑未知输出长度的不确定性。本轮没有确认论文作者仓库；同名 API 网关项目不能自动当作论文官方实现。

**你可以做什么。** 先复现输出长度分布估计与成本排序，在回放中对比 FCFS、SPF、单点长度预测和带不确定性的策略，再选一个最小调度切口迁移到 V1。

**非算子门槛。** 预测服务和 CPU 调度不需要新算子。训练/测试轨迹必须分开；记录预测开销、漂移、冷启动，以及低置信度时如何回退，不能让真实未来长度泄漏进在线策略。

**实验与交付。** 同时报告等待、公平性、SLO 与预测误差。先核对已有缓存感知准入与 SLO PR，不能将更换一个排序公式包装成全新系统。缺少作者代码使复现成本更高，暂列研究储备。

### P18：EvoSpec——从验证反馈在线适配草稿模型

**依据。** [论文 v3](https://arxiv.org/abs/2605.27390v3)联合调整活动词表与轻量草稿参数，包含异步 token 邻居检索和在线 LoRA。以该版本为准，本轮未确认作者代码。已有词表 PR 及 Speculators 在线训练工作需要进一步比较。

**你可以做什么。** 先分别复现“仅词表适配”和“仅在线 LoRA”两项，再考虑联合。明确收集哪些已验证数据、何时更新、如何绑定每轮候选使用的权重版本。

**非算子门槛。** 可以复用训练和推理算子，但训练与 CUDA Graph、并发请求、权重切换存在冲突。不能在仍有 in-flight batch 时无协议地原地更新权重。

**实验与交付。** 加入训练/更新延迟、额外显存、领域切换、接受率和多租户隔离的对照。只要保留精确目标验证也仍需验证具体采样实现。适合后续在线适配研究，第一版更可能贡献训练/适配组件，而非一个简单 proposer。

## 4. 固定源码中应该从哪里读

以下是**真实存在的阅读入口**，用于评估接入面；它们不是声称已经跑通的移植调用链，也不表示新方法只需改这些文件。行号均对应学习源码基线。

| 方向 | 路径、符号与行号 | 为什么读 |
| --- | --- | --- |
| 量化 | [vllm/model_executor/layers/quantization/base_config.py:97](../../vllm/model_executor/layers/quantization/base_config.py#L97)，`QuantizationConfig` | 了解配置、支持 dtype 与模块执行方法的契约 |
| 量化注册 | [vllm/model_executor/layers/quantization/__init__.py:50](../../vllm/model_executor/layers/quantization/__init__.py#L50)，`QUANTIZATION_METHODS` | 区分注册一个名称与实现有效加载/执行 |
| 投机入口 | [vllm/v1/worker/gpu/spec_decode/__init__.py:8](../../vllm/v1/worker/gpu/spec_decode/__init__.py#L8)，`init_speculator` | 确认固定版本实际创建哪些 speculator |
| 投机配置 | [vllm/config/speculative.py:72](../../vllm/config/speculative.py#L72)，`SpeculativeMethod` | 对照方法选择、校验与已有支持，不能仅因没有名称就判定无等价实现 |
| 候选预算 | [vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:36](../../vllm/v1/worker/gpu/spec_decode/adaptive_verification.py#L36)，`_assign_draft_token_budget` | 理解已有跨请求候选分配，避免重复实现 |
| 调度 | [vllm/v1/core/sched/scheduler.py:535](../../vllm/v1/core/sched/scheduler.py#L535)，`Scheduler.schedule` | 找到实际参与本轮的请求与 token 决策 |
| 输出状态 | [vllm/v1/core/sched/scheduler.py:1837](../../vllm/v1/core/sched/scheduler.py#L1837)，`update_from_output` | 跟踪已接受 token、拒绝及请求结束后的状态 |
| KV 生命周期 | [vllm/v1/core/kv_cache_manager.py:343](../../vllm/v1/core/kv_cache_manager.py#L343)，`allocate_slots` | 区分逻辑保留、物理分配和缓存命中 |
| 传输扩展 | [vllm/distributed/kv_transfer/kv_connector/factory.py:27](../../vllm/distributed/kv_transfer/kv_connector/factory.py#L27)，`KVConnectorFactory` | 评估外部缓存/预取协议是否能通过 connector 表达 |
| 执行与形状 | [vllm/v1/worker/gpu/model_runner.py:1563](../../vllm/v1/worker/gpu/model_runner.py#L1563)，`execute_model`；[vllm/v1/worker/gpu/cudagraph_utils.py:214](../../vllm/v1/worker/gpu/cudagraph_utils.py#L214)，`_init_candidates` | 验证策略是否改变实际输入、执行形状和 graph 选择 |

CPU 策略调用、进程消息、异步传输完成和 GPU 执行需要分别画出；不能将外部控制器与 target 的跨进程通信画成一个普通 Python 函数栈。进一步阅读已有[第七章](../chapters/07_processes_and_async.md)、[第九章](../chapters/09_speculative_decoding.md)和[第十章](../chapters/10_experiments_and_profiling.md)。

## 5. 已有工作与暂不另开同功能 PR 的方向

| 方向 | 已核实的重叠 | 当前处理 |
| --- | --- | --- |
| DFlash / DSpark / TurboQuant 首次集成 | 固定仓库已有相应实现入口 | 可以研究扩展，不能作为未接入的新算法 |
| LMCache 首次连接 vLLM | 固定仓库已有 connector；外部包是否安装另算 | 具体兼容/策略工作需查作者与上游两边 |
| 通用动态/缩小草稿词表 | [#59740](https://github.com/vllm-project/vllm/pull/59740)、[#60387](https://github.com/vllm-project/vllm/pull/60387) | NanoSpec/EvoSpec 必须解释方法差异，暂停重复模块 |
| MRV2 suffix proposer | [#60339](https://github.com/vllm-project/vllm/pull/60339) | SAM 必须证明相对现有检索路径的价值 |
| KV TTL / 范围保留 API | [#38514](https://github.com/vllm-project/vllm/pull/38514) | Continuum 走策略协作，不另造同功能 API |
| 独立 draft 服务与候选缓存 | [#42109](https://github.com/vllm-project/vllm/issues/42109)及其公开分支 | SSD 先审计现有原型 |
| 只给等待队列加 SLO 排序 | [#53571](https://github.com/vllm-project/vllm/pull/53571) | SLAI 聚焦运行中 decode 紧迫度，仍需复核差异 |
| 仅添加 EMA、冷却与滞回选 K | [#47111](https://github.com/vllm-project/vllm/pull/47111) | 不再作为独立新功能选题 |
| 新加 DDTree 或 FLy 模块 | [#42910](https://github.com/vllm-project/vllm/pull/42910)、[#53987](https://github.com/vllm-project/vllm/pull/53987) | 本轮发现已有开放 PR，不推荐重复提交 |

这些限制不妨碍学习和实验。某个项目暂不适合独立 PR 时，可以做复现、提出有证据的问题或参与已有工作；本轮没有代表你发送任何消息。

## 6. 统一的最小验证与性能实验

### 阶段一：先判断能不能做

为选中的方法写一页契约：输入、输出、持久状态、允许改变的语义、失败回退、依赖的算子和检查点。锁定作者代码 commit 与论文版本，审计代码、子模块、数据和权重许可；仓库公开不等于所有文件都可直接复制进 Apache-2.0 项目。

根据方向完成一个最小验证：量化验证单层矩阵与缩放，KV 压缩验证评分/保留集合/真实块回收，投机验证候选前缀与接受回退，调度验证一小段人工到达流与请求状态。验证不成立就调整范围，不继续堆功能。

### 阶段二：正确性分三类

| 方法性质 | 必须验证什么 | 不充分的证据 |
| --- | --- | --- |
| 调度、准入、通信与生命周期 | 没有 token 丢失/重复，请求结束和取消正确，资源最终释放 | 服务没有崩溃 |
| 声称无损的 token 级投机 | 贪心回归、EOS/拒绝边界、小词表分布测试与采样统计；检查 RNG 和数值差异 | 同 seed 的少数样例相同，或几个任务分数相近 |
| 量化、KV 剪枝、语义推理与文档拼接 | 先符合作者参考语义，再评估质量与性能 Pareto，报告质量退化 | 生成文本看起来通顺 |

无损是分布层面的论断，不必然意味着不同执行路径在同 seed 下逐 token 相同。差异要定位，不能一概归因于浮点误差。KV 剪枝要区分缓存中保存的字节、分配器可再次使用的块，以及显卡工具报告的进程显存。

### 阶段三：性能证据

1. 固定模型 revision、tokenizer、模板、输入 token、输出上限、采样参数、引擎 SHA、后端和 graph 模式；保存实际依赖快照。
2. 将校准/训练/索引/缓存准备、冷启动、预热和稳定运行分开计时；提供冷缓存与暖缓存结果。
3. 同时做低并发延迟和受控到达率实验。保持输出长度可比，记录截断和失败请求，不能通过少生成 token 制造加速。
4. 报告吞吐、TTFT、逐 token 延迟分布、P99、goodput、抢占/重算、实际缓存字节或块；Agent 项目加程序完成时间和工具等待。
5. 对不同负载、模型适配情况和资源压力做消融；重复独立进程运行。Laptop GPU 记录功耗、温度和频率，避免热状态变化主导结论。
6. 输出原始数据、分析脚本、失败案例和可复现命令。CPU 回放仅证明决策逻辑，不能作为 GPU serving 加速证据。

当前仓库的[实验章节](../chapters/10_experiments_and_profiling.md)已有基本命令与记录协议。上述步骤尚未为任何候选运行；本轮不提供性能数字。

## 7. 从项目走到 PR 与简历

建议首先做出能审查的最小实现与证据，再决定 PR 范围。完成算法复现并不自动说明应该合入核心仓库；中间件、量化工具链、Speculators、LMCache 或作者仓库可能更合适。

提交前按[AGENTS.md](../../AGENTS.md)再次查关联 issue、完整评论、引用该 issue 的开放 PR 与领域关键词。`gh` 在本机未找到，本轮使用 GitHub REST 工具做只读查询；不是已完成所有候选的最终 PR 准入检查。

以下是**未来选定题目后的命令模板，本轮未执行**；将占位符替换为实际编号与关键词：

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

Codex 可以辅助阅读、实现、调试和分析，你需要理解方法、逐行审阅、运行测试并能解释实验。仓库禁止纯代码代理 PR，要求披露 AI 辅助；文档自动推送授权也不等于自动创建上游 PR。

简历写法模板，只有实际完成后才能填写：

> 复现【方法与论文】，将其适配到 vLLM V1 的【模块】；完成【正确性/质量验证】及【模型与负载】实验，在【相同资源与质量约束】下将【指标】由【实测值】改善至【实测值】，公开复现材料并提交 PR【链接】。

“算法复现”“方法变体”“工程集成”分别说明；没有作者依据或查重证据，不写“提出全新算法”或“首次实现”。PR 未合并时如实标注状态。

## 8. 检索记录与外部版本快照

配套文件：[project_candidates_audit_2026-10-16.json](project_candidates_audit_2026-10-16.json)。它保存方法名查询、领域扩展查询、相关 PR 摘要、仓库 commit，以及检索局限。快照只说明本轮观察到哪个版本，不代表源码、许可或性能已经完整审计。

本轮对 18 个名称分别查询上游 issue 与 PR，PR 查询不限制状态；另外查动态词表、suffix automaton、position-independent、SLO scheduler、在线 draft/LoRA 和 kvpress 等领域关键词。领域搜索只读取返回的部分结果，不能据此证明整个领域无人开展工作。EPIC、CONCUR 等通用词有无关命中，保留原始命中计数，不把它们算作对应算法集成。

补充阅读了 SSD 相关 RFC #42109 的评论与分支链接，以及动态词表和 KV 保留的相关 issue/PR。其他候选未逐一读完所有关联评论；正式选题前必须补查。EvoSpec、SageSched、CONCUR、CacheScout 的作者代码均未确认，不能将同名第三方项目充当官方实现。

### 外部代码观察快照

下列 commit 由 GitHub API 返回，均为只读记录；没有下载、切换或安装这些版本。完整 SHA 见 JSON。

| 仓库 | 观察到的 commit |
| --- | --- |
| SINQ | [42ca83edf8fe](https://github.com/huawei-csl/SINQ/commit/42ca83edf8fe1afc48c40fca6acc973c5d2bb59f) |
| NanoSpec | [e836c00f5892](https://github.com/csAugust/NanoSpec/commit/e836c00f589215e1fb7e08413026c040e8ce3f80) |
| KVpress | [7136c224105b](https://github.com/NVIDIA/kvpress/commit/7136c224105ba7b046a6e538742c3589698680b8) |
| FastKVzip | [e04afaa6007c](https://github.com/Janghyun1230/FastKVzip/commit/e04afaa6007c58d18500d0698d2d8ae592ff22c2) |
| SAM-Decoding | [aaf939819223](https://github.com/hyx1999/SAM-Decoding/commit/aaf939819223147ce544e3a80f65758ac7758076) |
| SSD | [d7eb8fa0edb7](https://github.com/tanishqkumar/ssd/commit/d7eb8fa0edb77a6d0876af1903367b9bb82f54e7) |
| JetSpec | [2c7b3fae7569](https://github.com/hao-ai-lab/JetSpec/commit/2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f) |
| vLLM-JetSpec | [f90d5ca17a2c](https://github.com/JetSpec-project/vllm-jetspec/commit/f90d5ca17a2c05f436a80ee2e0984cc7a22e1a16) |
| SpecFit | [4973b3cdc48e](https://github.com/Chlience/SpecFit/commit/4973b3cdc48e5c016bc6c5ed18639d3163e8b2f5) |
| SpecReason | [6fd5b1292959](https://github.com/ruipeterpan/specreason/commit/6fd5b12929590e6824957191cc2e40a93d021339) |
| EPIC | [3204410f1723](https://github.com/DerekHJH/epic/commit/3204410f1723ed0f39575b4eba8c17578a1ee1a1) |
| KVLink | [41785d8e41f4](https://github.com/UCSB-NLP-Chang/KVLink/commit/41785d8e41f430671e2c080fbfb01e367b733ae2) |
| SLAI | [5098a7aba05e](https://github.com/agrimUT/SLAI/commit/5098a7aba05e3edbcfa3a509d6cc9cd248fc4380) |
| Continuum preview | [316a58794a6f](https://github.com/Hanchenli/vllm-continuum/commit/316a58794a6ff86b216e579b74fd56ed0c5a911f) |
| ThunderAgent | [7ddc8610270e](https://github.com/ThunderAgent-org/ThunderAgent/commit/7ddc8610270e56d3b109eed8796b3a4360fc67c9) |

### 本轮实际检查

本轮只验证文档结构、本地源码引用、链接目标、查询记录和格式；没有运行论文算法、模型质量评测、GPU 推理或性能分析。实际检查结果写入[检查记录](../checks.md)，提交与推送结果在交付消息中报告。
