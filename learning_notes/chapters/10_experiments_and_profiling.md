# 第十章：把源码理解变成正确性与性能实验

固定源码 commit 沿用首页。本章提供一个可执行的小实验闭环：在不改业务源码的前提下，对同一模型改变调度 token 预算，先比输出，再记录正式计时。当前环境仍缺推理依赖，因此交付的是协议与脚本，不是已经得到的性能收益。

## 1. 本章要回答的问题

1. 单元测试、模型输出对照、端到端性能分别证明什么？
2. 本版 benchmark 的计时和 token 分母在哪里？
3. 如何找出等待发生在 CPU、GPU 还是请求队列？
4. 一个足够小的实验怎样做到可重复、可归因？
5. 后续改投机预算时，什么证据才足以支持实习项目结论？

## 2. 先建立直观理解

性能不是一个脱离负载的常数。一次用户生成更快，与每秒服务更多请求，是两个实验问题。更大的批次可能提高设备利用率，也可能让某个请求等更久；同一组改动必须在明确的输入长度、输出长度、并发和资源下评价。

**benchmark**回答“指定窗口和工作负载里完成多少”；**profiling**回答“时间和资源花在哪里”。profiler 自身有开销，测得的 trace 有助于解释瓶颈，但不应直接替代无 profiler 的正式性能数据。

**正确性 oracle（对照依据）**先定义什么结果必须不变。当前教学实验使用相同模型的 greedy token ID 和停止原因精确对照；换成随机采样或算法近似后，需要重新设计分布／质量约束，不能照搬同 seed 就逐 token 一致。

## 3. 与 nano-vLLM 的对应关系

简化系统常先看总生成耗时和 tokens/s。过渡到 vLLM，需要补齐测试层级、warmup、调度参数、模型 revision、实际输出量、排队和图模式；否则很容易把模型初始化、不同输出长度或更多显存误认为系统优化。

## 4. 源码阅读地图

| 顺序 | 路径与符号 | 要核查的测量契约 |
| --- | --- | --- |
| 1 | [vllm/entrypoints/cli/benchmark/throughput.py:10](../../vllm/entrypoints/cli/benchmark/throughput.py#L10)，`BenchmarkThroughputSubcommand` | CLI 只是路由，真正计时在下层 benchmark |
| 2 | [vllm/benchmarks/throughput.py:37](../../vllm/benchmarks/throughput.py#L37)，`run_vllm`；第 1014 行 `main` | 模型创建、预热、生成与统计边界 |
| 3 | [vllm/benchmarks/throughput.py:1095](../../vllm/benchmarks/throughput.py#L1095)，实际输出统计分支 | 有输出对象时按实际 token 累计，不只按期望长度 |
| 4 | [vllm/benchmarks/serve.py:563](../../vllm/benchmarks/serve.py#L563)，`calculate_metrics` | TTFT／TPOT／ITL、完成量与 goodput 的定义 |
| 5 | [vllm/entrypoints/llm.py:783](../../vllm/entrypoints/llm.py#L783)，`LLM.start_profile`；第 793 行 `stop_profile` | profile 控制经前端和核心到 Worker，不只包用户 Python 栈 |
| 6 | [vllm/config/profiler.py:39](../../vllm/config/profiler.py#L39)，`ProfilerConfig` | 选 profiler 类型、输出目录与记录开销 |
| 7 | [examples/features/profiling/simple_profiling_offline.py:19](../../examples/features/profiling/simple_profiling_offline.py#L19)，`main` | 按当前示例看 start／stop 的实际用法 |

项目规则与测试方法见根 AGENTS.md；模型评测入口在 `tests/evals/`。本章不随意指定任意模型的 eval 配置为通用验证方案，应先按模型、输出变化和 serving 影响选择既有 suite。

## 5. 真实过程与时间窗口

一次离线实验包括：**CPU 初始化→模型加载／设备准备→warmup→正式 generate→核心多轮调度与 GPU 执行→结果回到 CPU→停止计时→保存结果**。第一章到第九章解释的整条链都位于正式 generate 内，而模型初始化与独立 warmup 应排除在稳态窗口外，并另外报告。

在线实验还增加客户端发送、网络、API 输入处理与排队。TTFT 包含哪些区段取决于请求计时定义；固定并发与固定到达率不能当同一个负载。只有离线 generate 总时间，没有请求级首 token 事件，就不能计算用户 TTFT 或 ITL。

profile 控制走 **前端普通调用→核心 client 消息→核心执行器→Worker**；GPU 活动由 profiler 捕获。共享内存、ZMQ、D2H 的等待必须按实际所在层解释，不能把一段 CPU 空白直接命名为“GPU 算力不足”。

## 6. 关键源码精读

### 6.1 总 token 与输出 token 是不同指标

`throughput.main`：

<!-- source: vllm/benchmarks/throughput.py:1125-1129 -->

```python
print(
    f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
    f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
    f"{total_output_tokens / elapsed_time:.2f} output tokens/s"
)
```

输入是测量窗口、请求数和 token 统计；输出只是报告，不修改请求状态。total tokens 包括输入与输出，output tokens 只包含实际生成。混用两者会让长 prompt 负载看起来拥有不相干的“生成吞吐”。还应回看实际输出与 fallback 期望长度统计分支，确认当前 backend 用了哪条。

### 6.2 TPOT 的分母不是所有输出 token 数

`serve.calculate_metrics`：

<!-- source: vllm/benchmarks/serve.py:614-620 -->

```python
tpot = 0.0
if output_len > 1:
    latency_minus_ttft = outputs[i].latency - outputs[i].ttft
    tpot = latency_minus_ttft / (output_len - 1)
    tpots.append(tpot)
# Note: if output_len <= 1, we regard tpot as 0 for goodput
all_tpots.append(tpot)
```

输入是单请求总时长、首 token 时长和有效输出长度。首 token 后的生成部分被分摊给其余输出；少于两个输出的请求有单独处理。ITL 是观测到的相邻流式事件间隔，投机一轮可返回多个 token，不能将某个事件间隔当成每个 token 的 GPU 计算时间。

## 7. 贯穿案例与最小改进实验

### 问题、假设与边界

沿第四章混合长短请求的案例，提出可验证问题：在相同模型和请求集合下，把 `max_num_batched_tokens` 从 32 改为 64，是否减少分块调度轮数、改善整个离线批次的完成时间？这是配置对照实验，不是新调度算法，也不预设 64 一定更好。

教学值 32、64 不是性能数字。实际派生的 `max_num_scheduled_tokens` 可能跟随配置变化，脚本会记录它；本实验评估的是这个用户配置旋钮及其派生行为，不宣称独立控制了每个内部字段。

固定条件：本地小模型（可沿用第一章 OPT-125M 快照）、单 GPU、MRV2、uni、greedy、eager、同步、启用 chunked prefill、关闭 prefix cache、相同 max_model_len、相同最大活动请求数、同 prompt 列表、相同输出上限和显存利用率。先使用小模型学会实验流程，再迁移到更有代表性的模型／负载；小模型结果不能外推到大模型。

### 实验工具与可执行命令

附带 [experiment_generate.py](../tools/experiment_generate.py)。`run` 真正执行推理并保存输入／输出 token、停止原因、实际输出数、每轮测量时长、依赖与硬件记录；已有输出文件会拒绝覆盖。`compare` 不导入 vLLM，可在没有 GPU 的环境检查两份实际记录。

**仅提供运行步骤，本轮没有生成任何真实性能 JSON。**前置条件与第一章相同，还需本地模型 revision 的真实记录。以下命令在仓库根目录的 WSL shell 执行：

```bash
export HF_HUB_OFFLINE=1
export LOCAL_MODEL=/absolute/path/to/local/opt-125m
export MODEL_REVISION=replace_with_actual_model_revision
mkdir -p learning_notes/artifacts
uv run --no-project --offline --python .venv/bin/python python learning_notes/tools/experiment_generate.py run \
  --model "$LOCAL_MODEL" --model-revision "$MODEL_REVISION" \
  --budget 32 --requests 8 --repeats 3 \
  --output learning_notes/artifacts/budget32.json
uv run --no-project --offline --python .venv/bin/python python learning_notes/tools/experiment_generate.py run \
  --model "$LOCAL_MODEL" --model-revision "$MODEL_REVISION" \
  --budget 64 --requests 8 --repeats 3 \
  --output learning_notes/artifacts/budget64.json
uv run --no-project --offline --python .venv/bin/python python learning_notes/tools/experiment_generate.py compare \
  learning_notes/artifacts/budget32.json learning_notes/artifacts/budget64.json
```

每个 `run` 新建模型实例，先做一次不计时 warmup，再测三个相同批次的 generate；耗时覆盖完整离线生成和结果返回，不包含初始化，也不提供 TTFT。脚本使用固定模板生成八条请求，记录实际 tokenization；输出上限不是保证输出长度，EOS 可能提前结束。

观察点：先要求重复之间以及两个配置之间 prompt IDs、output IDs、finish_reason 一致，然后比较每次实际 output tokens/s、离散度与批次时长。脚本会拒绝明显的配置／硬件／版本差异，但 GPU 功率状态、后台进程、温度等还要人工记录；model_revision 是调用者填写的标签，不等于脚本已经核实所有权重文件哈希。

脚本不记录 Scheduler 内部轮数。若要验证“分块轮数减少”这个机制假设，还需单独查看 iteration 日志或 trace；仅有批次 wall time 变化，最多证明该负载下的时间差，不能直接归因某段源码。

初步通过后至少交换方法顺序独立再跑，并使用新的输出路径；区分“三次同一实例重复”与“三次独立进程实验”。不要把一个 Python 进程的三个重复伪称为三个独立环境。结果在噪声内或退化时如实记录，不反复筛选样本。

### 从最小实验扩展到项目

| 层级 | 改什么／测什么 | 完成标准 |
| --- | --- | --- |
| 输入与状态正确性 | 既有单元测试、边界案例 | 解释契约，覆盖取消、抢占、长度、prefix 与投机拒绝等受影响行为 |
| 模型一致性 | 同输入的 greedy token 对照，适用模型 eval | 记录首个差异，不能只看文本近似；模型／serving 受影响时执行相关评测 |
| 离线性能 | 固定批次与实际输出数 | 有 warmup、原始记录、重复、条件和负结果 |
| 在线性能 | TTFT、TPOT、ITL、请求延迟、吞吐、失败 | 区分低并发、高并发与开放到达率；包含队列和失败计数 |
| 投机预算 | AR／原投机／均匀预算／待评估策略 | 分别记录 draft、select、pack、verify、state 与 padding 成本，检查公平性和接受定义 |

下一步真正改代码时，只选一个决策点，例如一处合法的预算选择，保留 off／baseline 路径；先确定调度、Runner 和 verifier 能表达，再设计最小测试。当前学习笔记阶段不做这个业务源码修改。

## 8. profiling 与已有 benchmark 练习

### 从可运行命令确认 CLI

**仅提供步骤**，运行依赖齐全后先读本版帮助：

```bash
uv run --no-project --offline --python .venv/bin/python python -m vllm.entrypoints.cli.main bench throughput --help
uv run --no-project --offline --python .venv/bin/python python -m vllm.entrypoints.cli.main bench serve --help
```

然后才按帮助和已检查的 `add_cli_args` 选择模型、数据、长度、profile 与输出参数。尽量复用已有 benchmark 扩大负载；本章的小脚本用于记录可核对的最小离线案例，不能替代服务压测。

### 用时间线定位原因

在无 profiler 的基线已经运行成功后，单独采集小规模运行：

```bash
nsys profile --trace=cuda,nvtx,osrt --output=learning_notes/artifacts/budget_profile \
  uv run --no-project --offline --python .venv/bin/python python \
  learning_notes/tools/experiment_generate.py run \
  --model "$LOCAL_MODEL" --model-revision "$MODEL_REVISION" \
  --budget 32 --requests 8 --repeats 1 \
  --output learning_notes/artifacts/profile_only.json
```

本轮未执行。trace 包含加载和预热，需要在查看器中选择正式生成范围；profile_only.json 仅辅助定位，不能混进无 profiler 的正式性能汇总。另可沿当前 `ProfilerConfig` 和官方仓库示例使用 PyTorch profiler 的 start_profile／stop_profile；停止时等待实际文件写出。

按证据排查：CPU launch 间隙长则查输入组织／Python／同步；GPU attention 时间长则核对上下文与 KV 工作量；算子很忙但请求 TTFT 差则查排队与调度。每条只是排查假设，需要 trace 和对照确认，不能从一张截图就断定瓶颈。

**无 GPU 替代且本轮可做**：运行引用检查、查看实验脚本 `--help`、检查 compare 对相同／不同 token 记录的行为，并纸面设计实验矩阵。合成记录只验证比较逻辑，绝不能充当 GPU 数据。`artifacts/` 已忽略，避免自动推送原始 trace；没有生成的文件不会被提交。

## 9. 易错点与自我检验

1. 两组测试设置相同 max_tokens，为什么实际输出工作量仍可能不同？
2. 为什么先比较 token，再分析性能？
3. 同实例重复三次与独立启动三次有什么区别？
4. 离线脚本能报告 TTFT／P99 吗？
5. acceptance 提高但端到端变慢，应该记录哪些成本？
6. 输出一致性失败能否直接归因为浮点误差？
7. 为什么实验记录必须包含实际派生预算、Runner 和 graph 条件？
8. 没有 GPU 时什么产物能交付，什么结论必须保留未验证？

## 10. 本章总结与项目接续

现在应能定位本版 benchmark 的计时与统计、定义输出对照、使用小实验脚本收集可复查记录，并说明 profiler 与正式测量的区别。章节编写和静态核对已经形成闭环，运行环境、模型评测和性能收益仍需实际执行，不能因文档齐全就标项目完成。

后续实习项目可沿第九章选一个预算决策点：先调查已有实现与研究，明确正确性约束和支持组合，再修改、复用既有测试、做模型评测，最后复用本章测量协议。提交 PR 前按 AGENTS.md 查重、由人逐行审阅并披露 AI 协助；当前只是将学习资料推送到用户 fork，不自动创建 PR。

### 自我检验答案

1. EOS／stop 可能提前结束，必须记录实际输出数、原因及输入 IDs。
2. 性能对照应建立在相同有效任务和正确输出上，否则耗时差不可归因。
3. 前者共享模型实例、缓存和热状态；后者包含重新建立状态的独立过程，二者都需明确窗口。
4. 不能，脚本没有逐请求流式事件，也没有足够服务负载；仅报告离线窗口。
5. 草稿、选择、打包、目标验证、状态维护、D2H／同步和 graph padding，以及不同负载。
6. 不能，要定位首个差异、比较 logits 与路径、排查状态和数值问题，保留失败证据。
7. 用户参数可能触发派生配置或不同执行分支，否则对照并非只改变宣称的因素。
8. 能交付源码地图、手算账本、可执行协议和静态检查；不能交付虚构的测试通过、模型一致性或 GPU 性能数字。
