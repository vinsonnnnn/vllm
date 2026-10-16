# vLLM + DFlash + D-Cut：技术方案与 Codex 执行提示词

版本：2026-10-16。用途：交给本地或远程开发环境中的 Codex，逐阶段实现、验证并评测。本文是实施方案，不是已完成的实验报告。尚未检查你的 GPU、仓库或运行任何 vLLM 实验。

## 1. 项目目标和边界

项目名称建议：基于 vLLM 的负载感知投机解码验证预算优化。

目标：在固定 vLLM commit、固定目标模型和匹配的 DFlash 草稿权重上，先复现 D-Cut，再探索减少策略开销、改善成本预测或执行桶选择的扩展。在正确性成立的前提下，测量真实端到端收益与退化边界。

三类贡献必须区分：
- 上游能力：vLLM、DFlash 的已有实现。
- 复现与移植：D-Cut 论文和公开实现的适配，不宣称原创算法。
- 个人扩展：有基线、消融、代码和测量支持的具体改动。研究过后才判断是否有新颖性。

第一版范围：单节点、单 GPU、TP=1、文本生成、greedy、同步调度。使用已有草稿权重，不训练草稿，不做树解码、PD 分离、多 LoRA 或多模态。模型优先从有匹配权重的 Qwen3-4B 开始；显存足够再扩到 8B。具体模型 ID、revision、草稿块大小与兼容性必须检查模型卡和当前源码，不能按显存大小猜支持情况。

## 2. 已核查事实与资料

截至本次核查：
1. vLLM 官方文档列出 dflash 相关配置，同时已有 Dynamic Speculative Decoding 和 Adaptive Verification 路径，后者文档注明当前针对 DSpark。必须检查与本项目的重叠。
2. D-Cut 公开 vLLM PR #47131 仍为 Open，不应认为已经合入稳定版。分支可能存在冲突，使用前记录 PR 的 base/head SHA，并检查讨论、diff 与测试。
3. PR 描述的兼容范围是 V1 runner、piecewise CUDA Graph；它对异步调度、full graph、Model Runner V2 设有限制。其描述不是对未来任意版本的兼容承诺。
4. PR 方案是在 DFlash 已经打完草稿后裁剪候选，主要减少目标验证成本；并不直接减少这一轮 DFlash 草稿主干计算。

资料入口（执行时重新核查，不把 latest 文档当固定版本规范）：
- vLLM：https://github.com/vllm-project/vllm
- 投机解码文档：https://docs.vllm.ai/en/latest/features/speculative_decoding/
- DFlash 论文：https://arxiv.org/abs/2602.06036
- DFlash 仓库：https://github.com/z-lab/dflash
- D-Cut 论文：https://arxiv.org/abs/2607.14647
- D-Cut 技术说明：https://angelslim.readthedocs.io/zh-cn/latest/dcut.html
- D-Cut PR：https://github.com/vllm-project/vllm/pull/47131
- PR diff：https://github.com/vllm-project/vllm/pull/47131/files

## 3. 分阶段实施方案

### P0：仓库、硬件和可行性审计

输出 environment.md、source_map.md、upstream_audit.md、manifest.json。

检查：git status/分支/SHA、AGENTS.md、Python/PyTorch/CUDA/驱动、GPU 型号/显存、CPU/内存/磁盘、依赖安装方式、实际导入 vLLM 路径、模型缓存与 GPU 可用性。无需读取或输出账户凭证。

固定兼容基线 SHA，创建独立本地分支或 worktree，保留用户已有改动。记录目标/草稿模型 revision、tokenizer、dtype、attention backend、最大长度、草稿长度、调度和 graph 配置。不能同时追踪 main 并持续升级依赖。

沿当前源码建立真实路径：配置解析 → DFlash proposer/目标特征 → 草稿结果传递 → scheduler 应用 → runner 输入打包 → verifier/rejection sampler → 请求状态与 KV 更新。文件名、类名与字段必须通过检索确认。

若无 GPU，可完成审计、纯逻辑实现和测试，GPU 步骤明确标成 BLOCKED；不能用 CPU 或 mock 耗时充当 GPU 性能结果。无需为本任务租用或购买任何资源。

### P1：建立 AR 与 DFlash 基线

先运行最小单请求，再运行多请求。AR、DFlash 使用同一个目标模型、采样和工作负载；DFlash 同时加载匹配草稿模型。greedy 先检查输出 token ID；明确 thinking/non-thinking 模式和 chat template。

保留未修改运行日志、命令、版本、输入 token ID 和计时范围。用现有 benchmark 工具优先，缺失的记录项再补充。DFlash 本身不正确或不兼容时，先修复或选择有证据支持的兼容 commit，不能跳过基线直接做剪枝。

### P2：复现 D-Cut

实现四个可切换模式：off、uniform、confidence_fixed_budget、confidence_cost_model。命名是本地设计建议，不是假定上游存在的参数。

- off：完全走原 DFlash 路径，不增加热路径策略计算。
- uniform：相同预算尽量均匀分给请求，是公平对照。
- confidence_fixed_budget：同一批次跨请求分配预算。
- confidence_cost_model：枚举少量预算，按实测成本与预计推进量选择。

逻辑接口建议（先适配当前架构，不能强行照抄签名）：

```python
select_keep_lengths(
    draft_scores, valid_draft_lengths, request_ids,
    budget, runtime_features, cost_profile,
) -> keep_lengths
```

要求：keep_lengths[i] 是请求 i 保留的候选前缀长度，0 <= keep_lengths[i] <= valid_draft_lengths[i]。用 request ID/轮次等信息保证结果对齐，不能依赖请求列表永不重排。只保留前缀，不挖洞保留后面的 token。

第一版定义：n 是参与本轮验证的有效请求数；k_i 是保留的候选数；A 是候选预算，则 sum(k_i) <= A。教学模型中目标验证 query 数近似 Q = n + sum(k_i)，n 表示每请求的基础验证位置。实际 anchor/bonus、跨轮 token 和 prefill 混合语义必须从源码核实。文档、实现与统计统一约定，不能把 bonus、anchor 或已提交 token 重复计数。

候选分数：从真正选中草稿 token 的 log probability 构造前缀累计分数，再选高收益前缀；使用 log-domain 避免长前缀乘积下溢。它只是接受收益的代理信号，不是未经校准就成立的接受概率。最终公式以论文、PR 与 verifier 语义为准，说明索引的 off-by-one。

全局选择要满足前缀约束、预算约束、padding 无效位置屏蔽和确定性 tie-break。若使用 top-k，要证明并测试这些性质。NaN/Inf、空批次、部分序列剩余输出长度不足时有明确策略，不能传播脏数据。

链路：草稿生成 → 分数/预算决策 → 按请求传递长度 → scheduler/runner 真实减少验证输入 → 原 verifier → 正确更新目标 KV、草稿 KV、hidden states 与序列状态。

禁止仅丢弃输出 token 或屏蔽 loss 就声称节省计算。必须确认实际执行的验证 token、tensor shape 与 kernel work 确实发生变化。验证额外工作是否被 graph padding 抵消。

兼容策略：先沿用参考实现可支持的同步/piecewise 路径；明确禁用不兼容组合并给出错误信息，不能默默改变用户配置。参考 PR 的 dflash_dcut 参数只在该实现中有意义，不得直接认为本地版本支持。

### P3：只选择一个有测量依据的扩展

优先候选 A：降低选择开销。profiling 证明 prefix-score、top-k 或 CPU 同步是瓶颈后，才实现 GPU 驻留张量、融合扫描/选择或减少同步。先使用 PyTorch 正确实现作为 oracle，必要时用 Triton；不要为增加技术栈强写 kernel。

优先候选 B：更好的成本模型。先用 (有效 decode 请求数, 候选预算) 建表，再研究上下文长度分布、总有效验证 token、graph bucket 对预测误差的影响。用独立校准负载，测量整个 draft/select/verify 周期而非只计目标 GEMM。profile 缓存键至少包含硬件、目标/草稿模型 revision、dtype/backend、TP、graph 模式、关键 vLLM SHA 与形状设置；缓存失配必须重建或明确回退。

候选 C：graph-aware 预算。仅在固定版本确实支持相关 graph 形状时比较合法桶；先验证 piecewise，再探索 full graph。若 padded shape 相同且计算量未减少，不能从逻辑 token 数减少推导加速。不要直接去掉兼容检查就称支持 full graph。

SLO/公平性是后续扩展：先实现稳定的性能路径，再加入等待时间、请求老化与最低进展约束。若实验范围不足，不同时加多个策略，以免收益归因不清。

候选目标可写为：max 预计本轮提交 token 数 / (T_draft + T_select + T_pack + T_verify + T_state)。所有项是示意模型，决定时不得使用本轮验证之后才知道的接受结果。只有一轮内不随预算变化的常数项也会影响预算之间的比值，不能任意删去。

### P4：正确性与回归测试

纯逻辑：合法范围、总预算、前缀闭合、空批次、k=0/全部保留、无效 padding、tie/NaN、request 重排、成本表失配、off 热路径、uniform 与 confidence 相同预算。

GPU 集成：AR/DFlash/off/各剪枝模式的 greedy token ID 对照；不同提示长度、max_tokens 边界、EOS、首候选拒绝、全部接受、不同接受长度、block 边界、新请求加入、结束/取消、抢占恢复（若本版本支持）、重复轮次运行后 KV/hidden state 一致性与无泄漏。小样本正确性跑通后再扩到 100 条以上多类输入。

若 token 不一致，报告首个差异位置、对应 logits/差距、执行模式和复现命令。区分基线数值变化与引入的状态错误；不得直接归咎浮点误差。不能通过修改目标采样规则、增加近似容忍、静默丢失败样本来通过。

随机采样不是第一版默认支持。草稿依赖的自适应裁剪是否影响 proposal 分布与 rejection sampler 假设，必须单独分析；保持代码表面 sampler 不变并不足以证明无偏。需要算法论证、小词表枚举/采样收敛测试和 GPU 集成测试。相同 seed 输出一致不是证明。未验证时显式限制 temperature=0，不宣称通用随机采样无损。

性能测量版本关闭逐步同步、详细 tensor dump、每请求 verbose 日志等调试设施；统计应批量异步汇总，避免观测本身制造瓶颈。

### P5：公平的性能实验

方法组：AR、DFlash(off)、uniform、D-Cut fixed-budget、D-Cut cost-model、个人扩展。已有上游同类方法可用且兼容时加入；不可用记录原因。不能只击败一个很差的固定长度。

两条对照轨道：
1. 控制变量：同一 piecewise/eager 兼容模式、相同资源上限与参数，隔离剪枝本身收益。
2. 实际部署：各方法使用合法的较优设置，比较 AR/full graph 与剪枝/piecewise 等真实组合，完整披露差异。

资源公平：记录实际 KV block 数、草稿显存、graph pool 和峰值显存。AR 不加载草稿，自然可能拥有更多 KV 容量；既报告同 GPU 的真实部署结果，也在需要时增加同 KV 容量控制实验。不能只设相同 gpu_memory_utilization 就宣称内存完全相同。

先 smoke，再筛选：并发 1/4/8/16/32（显存允许再 64），输入约 128/1024/4096 tokens，输出约 128/512 tokens。不要首次全组合。选择 3 个代表负载完成核心矩阵，再扩展确认边界。精确长度以 tokenization 结果为准。

工作负载：数学、代码、对话三类，先校准集，再独立评测集。记录数据集版本、样本 ID、抽样 seed、prompt/template、thinking 模式。固定长度合成测试可 ignore_eos，但必须对所有方法相同且单独标注；真实任务测试保留 EOS，并核查任务质量。

封闭并发测试与开放到达率测试分开。后者先用 baseline 探索饱和区，再固定几个负载强度和同一 arrival trace/Poisson seed 比较；包含排队时间，记录实际发送时刻与计划时刻，监测压测客户端饱和。先固定长度，再混合长度、突发和长短请求混合。

计时：独立 warmup 与正式测量；编译/graph capture/建表不混入稳态，但单独报告冷启动开销。小规模筛选后，最终代表场景至少 3 次独立重复，随机或轮换方法顺序，禁止几组同时争用 GPU。报告每次结果与中位数/离散度；只有 3 次时不要伪装成高置信度结论。尾延迟需要足够请求量，样本不足则不作强结论。

指标：
- output tokens/s、requests/s、TTFT、请求级 TPOT=(完成时刻-首 token 时刻)/(输出 token 数-1)，少于 2 tokens 单独处理。
- ITL 与 TPOT 分开。投机解码可能一轮提交多个 token，流式事件间隔不能当作逐 token GPU 耗时。
- P50/P95/P99、失败/OOM/超时/取消数、实际输出量、显存。
- 生成候选数、保留候选数、验证 query 数、接受草稿数、提交 token 数、轮数，明确每项分母；特别区分每轮接受草稿和含 bonus 的总推进量。
- draft/select/pack/verify/state 分解、graph padding、同步次数（可测时）。profile 单独运行，不把 profiler 下耗时直接当正式吞吐。
- 若评估 SLO：提前在 calibration 配置中冻结 TTFT/TPOT 门限，同时满足两者的成功请求数/总测量时间定义 request goodput；分母包含所有已提供请求，失败不能隐去。尾部排空计时规则事先固定。

结果解释：无投机可比性、输出数不一致、debug 开销污染、只改善 acceptance 不改善 wall clock，都不能写成加速。允许得到负结果。建议预设工程筛选门槛：代表负载 >5% 相对最佳已有可用基线改善且超过测量噪声才作为显著优化候选；这只是筛选目标，不是性能承诺。正确性必须先过。

## 4. 目录与可复现产物

复用仓库现有约定，以下为建议，不必机械创建所有路径：

```text
docs/dflash_dcut_project/
  environment.md
  source_map.md
  upstream_audit.md
  design.md
  experiment_protocol.md
  progress.md
  final_report.md
benchmarks/dflash_dcut/
  configs/
  run_matrix.py
  summarize.py
tests/...
artifacts/dflash_dcut/<run_id>/
  manifest.json
  commands.txt
  raw_requests.jsonl
  engine_metrics.jsonl
  summary.json
  profiler/  # 不进入一般结果汇总
```

原始模型权重、token、个人路径与大体积 traces 不提交 git。每个实验有 baseline SHA、patch SHA/diff hash、模型 revision、配置、数据、退出码与时间戳。由原始数据自动生成报告，禁止手写数字。保留第三方许可证与来源。

## 5. 总控提示词：复制给 Codex

```text
你是本项目的实现工程师。请在当前工作区的 vLLM 仓库中，实施《vLLM + DFlash + D-Cut：技术方案与 Codex 执行提示词》；先通读该 Markdown 文档，再开展工作。

目标：固定兼容 vLLM commit，建立 AR 与 DFlash 基线，复现 D-Cut 的跨请求验证预算，然后根据 profiling 选择一个小范围扩展，完成正确性和端到端性能实验。不要只提供设计、解释或启动命令，要在授权环境中实际修改、测试并交付结果。

权限与边界：
- 允许创建本地分支/worktree、编辑项目源码和文档、添加相关测试、在隔离环境安装必要依赖、运行已有本地 GPU 和模型的实验。
- 先检查 AGENTS.md、git status、GPU、磁盘、依赖与模型；保留用户已有改动。不要修改系统驱动、系统 CUDA 或全局环境。
- 不租用/购买资源，不下载未经预算确认的大模型，不上传代码/日志，不自动推送或创建 PR，不访问与任务无关的凭证。
- 模型缺失时先报告精确模型 ID/revision、估算下载与磁盘需求、是否需要许可；继续做不依赖模型的审计、实现和纯逻辑测试。新增付费或大型下载需要我决定。

技术纪律：
1. 从当前源码确认 DFlash、Dynamic SD、Adaptive Verification、runner、graph 与 D-Cut 现状。读取论文和 PR #47131 的 diff/讨论，记录 base/head SHA；不能假设未合并 PR 已存在。若已存在等价实现，复用并评估，禁止重复添加。
2. 不训练草稿。第一版单 GPU、TP=1、文本、greedy、同步调度。使用匹配且可运行的目标/草稿权重。先解决基础正确性，再优化。
3. D-Cut off 必须保持原路径；增加 uniform、confidence fixed-budget、cost-model 模式，并明确配置是现有字段还是本地新增字段。
4. 只能裁剪候选前缀。核实 anchor/bonus/query/输出 token 与两套 KV、hidden state、跨轮请求映射；测试长度边界、EOS、拒绝、全部接受、请求进出和 block 边界。
5. 确认真正减少 target verify 输入和执行工作，不能仅在输出阶段裁剪。策略成本、CPU/GPU 同步、graph padding 都计入结果。
6. 先沿参考实现支持的 piecewise/eager 路径运行；不兼容 async/full graph/runner 组合显式校验。未证明 full graph 支持时不删除保护。先检查当前 commit 是否已消除这些限制。
7. 随机采样超出第一版；未证明草稿依赖裁剪与采样算法兼容前，显式限制 greedy。不能以相同 seed 或不改 sampler 源码宣称无损。
8. 先用原生张量实现 oracle；profiling 证明瓶颈后才写 Triton/CUDA。每项扩展必须能开关、能消融，并与已有最好可用基线比较。

实验纪律：
- 固定 SHA、模型 revision、prompt/template、dtype/backend、数据集/样本/seed 和资源设置；输出 manifest、完整命令、原始请求与引擎日志。
- 对照 AR、DFlash、uniform、D-Cut fixed/cost、扩展。分同执行模式的控制对照与各自合法最优设置的部署对照，记录 KV 实际容量和草稿/graph 显存。
- 先 smoke，后小矩阵，再对代表负载做至少三次独立重复。校准集与测试集隔离。不得后台无限压测或同时争用 GPU。
- 同时报告吞吐、TTFT、TPOT、尾延迟、失败、显存、接受长度与分阶段耗时。开放到达率与固定并发分开，统计队列等待，失败不从分母删除。
- 不得用 mock/CPU 耗时冒充 GPU 测量，不得将调试/profiler 结果当正式性能。没有 GPU 时明确 BLOCKED，并交付其他已完成产物。
- 收益不成立时保留负结果，分析是否来自选择开销、padding、数值/状态错误或计算瓶颈转移；不要反复筛样本直到得到好数字。

执行：先完成 P0-P1，并在可行时直接推进 P2 和正确性测试；不要停在计划或等待普通代码改动确认。P3 根据实测只选一个扩展。每阶段更新 progress.md（完成/失败/未测、命令、文件、结果和下一步），长期任务按阶段检查点结束，避免一个命令无上限运行。遇到权限、资源费用、必须下载大模型或硬件缺失时，停止依赖步骤并清楚报告；不依赖该资源的工作继续。

最终交付：可审阅 diff、配置/运行脚本、测试与原始结果、自动汇总表/图、报告和复现命令。说明上游能力、D-Cut 复现与个人扩展各是什么；实测与未测分别是什么。不得保证性能，不编造数字。
```

## 6. 分阶段提示词：需要控制节奏时使用

### 提示词 A：审计与基线

```text
请先通读技术方案，执行 P0-P1。核查本地仓库与 AGENTS.md，记录环境/基线 SHA，检查实际导入路径，定位真实 DFlash 与验证调用链，审查 D-Cut PR 和等价上游实现。创建独立工作分支，保留已有改动。
若模型和 GPU 可用，实际完成 AR/DFlash 最小正确性和小规模性能 smoke，保存输入 token、完整命令、原始日志与 manifest。若不可用，明确缺什么，完成其余审计，不伪造结果。此轮不做自创优化，也不进行大型下载。
交付 source_map、upstream_audit、environment、smoke 结果与下一阶段的最小改动设计。
```

### 提示词 B：复现 D-Cut

```text
继续既定 SHA 与环境，读取 progress.md 与审计结果。实施 P2，不重新升级依赖或生成无关架构。
按论文和已审查 PR 复现 off/uniform/confidence fixed-budget/cost-model，解释预算、anchor/bonus、请求映射和跨轮 KV/hidden state 语义。off 不引入额外热路径计算。使用现有 verifier，不改变目标采样规则。
完成前缀闭合、预算、重排、边界、配置与 off 回归测试，以及可用 GPU 上的 greedy 集成正确性。提供实际执行形状/验证 query 数，证明剪枝发生在目标验证前。先解决失败，再跑较大 benchmark；未测项标注清楚。
```

### 提示词 C：定位并实现一个扩展

```text
读取已有正确性报告和 smoke 结果，执行 P3。先 profile draft/select/pack/verify/state 与 CPU 同步、graph padding，输出证据。选择一个能解释收益的扩展：减少策略开销、改善成本模型或合法的 graph-aware 预算。
先查是否已有等价功能；不得把参考 PR、普通参数搜索或移植描述成原创。保持可开关，保留正确性 oracle。若成本主要来自 target 而非策略，不能凭喜好写选择 kernel。若动态预算无法减少实际 padded 执行工作，明确报告并调整合法候选预算，不虚构加速。
实现一个扩展、回归测试、消融和小规模实际收益验证。没有收益也如实交付，不同时叠加多个未验证功能。
```

### 提示词 D：最终实验

```text
执行方案 P5。先冻结实验协议与配置，在校准集上选定成本表和参数，然后固定测试集、输入/输出预算和 arrival trace。
测 AR、DFlash、uniform、D-Cut fixed/cost、个人扩展；分别给出控制对照和真实部署对照。先完成小矩阵，再将最终代表场景独立重复至少三次。逐组运行，不争用 GPU。
保存原始请求、失败/超时、完整命令、模型 revision、SHA、KV/显存设置及每轮关键统计；由脚本生成汇总。报告 output tokens/s、TTFT、TPOT、P95/P99（样本足够时）、失败率、显存、冷启动开销及消融；区分 token 提交与流式事件。
列出相对 AR、原 DFlash、原 D-Cut/最好已有基线的收益，不只挑最好一项。出现正确性差异就暂停相关性能结论，先定位。不能使用论文数字填本地报告。
```

### 提示词 E：最终审阅

```text
以代码审阅者身份检查最终 diff 与全部结果，不扩展功能。重点找：预算 off-by-one、anchor/bonus 重复计数、非前缀选择、请求重排错位、KV/hidden state 污染、隐式 graph 降级、策略中的同步、测试集调参、失败样本隐藏、不公平内存对照和统计口径错误。
将问题按严重性列出，修复实际缺陷并仅重跑受影响检查。给出真实可复现的最终报告、限制、负结果和复现命令。未达到收益目标则明确写“未证明性能改善”。简历描述仅使用已测且可归因的数字，并注明开源复现与独立贡献。
```

### 提示词 F：跨会话继续

```text
继续此项目。先读方案、progress.md、环境 manifest 和当前 git diff，复用已确认的基线 SHA、模型与结果，不重新从零开始。概括已完成/失败/未测，再执行下一项有依赖依据的任务。不要重复已完成的大规模实验；只对新改动、失败或未解决问题补测。
```

## 7. 完成判据

- 可复现：干净环境按记录能构建并运行；模型和代码版本明确。
- 正确：greedy、状态与边界测试成立，off 与原路径回归通过。
- 有证据：真实 GPU 数据、失败数据、对照、消融与噪声范围都保留。
- 有归因：指出收益或退化来自什么，不能只报 acceptance。
- 有边界：未支持功能显式限制，无 GPU 或未完成的随机采样不冒充支持。
- 有个人工作：移植、profiling、扩展与上游能力分别说明；不以 DFlash/D-Cut 名称冒充原创。

推荐执行顺序：审计和基线 → D-Cut 复现与正确性 → 一个测量驱动扩展 → 公平实验 → 审阅与报告。不要同时训练草稿、换模型、上多卡和改采样算法。
