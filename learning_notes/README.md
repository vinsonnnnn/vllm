# vLLM 源码学习笔记

面向已经读过 nano-vLLM、理解 KV Cache 与连续批处理、希望独立修改推理系统核心模块的读者。学习成果应当是：能从一次请求沿真实调用链定位状态变化，提出边界明确的修改，并用正确性检查和可复现的性能实验评价它。

## 版本与环境首页

核实日期：2026-10-16（Asia/Shanghai）。以下为本次实际检查的结果。

| 项目 | 固定记录 |
| --- | --- |
| 仓库目录 | Windows `E:\vllm`；WSL 对应 `/mnt/e/vllm` |
| 分支 | `main` |
| 完整 commit | `f9c98bd1ebbddb9acedb1e6d7505ad9588f9a266` |
| commit 时间 | 2026-10-05 11:17:01 +0800 |
| commit 标题 | `docs: add personal vLLM deployment guide` |
| origin | `https://github.com/vinsonnnnn/vllm.git`（fork） |
| 版本标识 | `git describe --tags --always --dirty` 为 `f9c98bd1e`；HEAD 没有 tag；不推断发布版本号 |
| Python 包版本 | 未安装 vLLM；`vllm/_version.py` 不存在；包版本通过构建动态产生，不能当作某个正式 release |
| 初始工作区 | `git status --porcelain=v1` 输出为空 |
| 当前命令入口 | Windows PowerShell，Windows NT 10.0.22631.0 |
| 可运行的 Linux 环境 | WSL2 Ubuntu，内核 `6.18.33.2-microsoft-standard-WSL2`，x86_64，glibc 2.43 |
| 现有虚拟环境 | `/mnt/e/vllm/.venv/bin/python` → `/home/vinson/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12`；Python 3.12.15 |
| uv | Windows 0.12.7；WSL 0.12.23，`/home/vinson/.local/bin/uv` |
| 已安装的相关包 | `pre-commit==4.6.2`；`vllm`、`torch`、`triton`、`transformers`、`pytest` 均无安装元数据 |
| GPU | Windows 与 WSL 的 `nvidia-smi` 均识别 NVIDIA GeForce RTX 4090 Laptop GPU；驱动 610.47；显存 16376 MiB |
| 分析工具 | WSL `/usr/local/bin/nsys`：Nsight Systems 2025.6.3.541；WSL PATH 未找到 `ncu`、`nvcc` |
| hooks | `.git/hooks/pre-commit` 存在，解释器指向 `/mnt/e/vllm/.venv/bin/python`；未据此认定所有 hook 环境已就绪 |

**能力结论：**现在能做源码检索、标准库脚本检查和手工调度推演。仓库 CPU 测试收集已实际尝试，因 `No module named pytest` 失败；这不表示测试断言失败，也不表示 CPU 后端没有实现。GPU 推理未运行：现有环境缺少运行依赖。`nvidia-smi` 成功不等于 PyTorch CUDA 或 vLLM 可运行。Nsight 仅验证了版本命令，未采集推理 trace，未验证 CUPTI 权限；不能报告吞吐、延迟或 kernel 时间。

原生 Windows 运行支持限制见 [docs/getting_started/installation/gpu.md:27](../docs/getting_started/installation/gpu.md#L27)。后续运行实验使用 WSL/Linux。Windows 显示为零字节的 `.venv/bin/python` 实际是 Linux 符号链接，已在 WSL 验证；不要在 Windows 重建或覆盖这个环境。

学习笔记编写期间没有安装、同步或升级推理依赖，没有下载模型，没有切分支。随后自动提交的 pre-commit 尝试初始化独立 hook 工具缓存，已停止并改为直接调用已有检查工具，详情见检查记录；项目 `.venv` 的推理依赖未变。解释器可用不代表依赖已按此 commit 构建。保留已有的 `笔记/vLLM从零部署指南.md`，没有将其当成当前运行成功的证据。

## 固定版本的约定

源码基线始终是上述完整 commit。学习笔记的提交会推进分支 HEAD；后续每轮确认基线仍是 HEAD 的祖先，且业务源码和开发配置与基线一致，再沿用源码行号。检查脚本仅排除 `learning_notes/` 与用户另行提交的 `优化/vllm_dflash_dcut_plan_and_prompts.md`；后者是项目方案文档，不是源码或已经执行的任务。若有其他差异，先指出偏差，不把新源码结论混入本套笔记，不自动切回旧 commit，不覆盖用户改动。

用户已授权：每轮更改并完成必要检查后，自动提交本轮自己的改动并推送到 `origin`（`vinsonnnnn/vllm`），本系列当前分支为 `main`；不夹带用户其他改动、不强制推送。将推送结果报告给用户。这项授权不包括创建 PR、切换源码版本或安装、升级依赖。

没有用户要求，不执行依赖安装、升级、环境重建或 Git checkout。将来补齐环境时，需要单独记录实际依赖快照、构建方式、模型 revision 和硬件条件；固定源码 commit 并不能自动固定整个实验环境。

## 目录与阅读顺序

| 章 | 主题 | 当前状态 |
| --- | --- | --- |
| 01 | [仓库地图与最小运行示例](chapters/01_repository_and_minimal_run.md) | 已编写并核对引用 |
| 02 | [单个文本请求从输入到输出](chapters/02_request_lifecycle.md) | 已编写并核对引用 |
| 03 | [Request、跨进程消息与输出状态](chapters/03_request_state.md) | 已编写并核对引用 |
| 04 | [Scheduler、两类 token 预算与 Continuous Batching](chapters/04_scheduler_and_budget.md) | 已编写并核对引用 |
| 05 | [KV Cache、Block Table、Slot Mapping 与 Prefix Cache](chapters/05_kv_cache.md) | 已编写并核对引用 |
| 06 | [Model Runner V2、输入组织与 Attention Backend](chapters/06_model_runner.md) | 已编写并核对引用 |
| 07 | [进程、通信、异步调度与执行重叠](chapters/07_processes_and_async.md) | 已编写并核对引用 |
| 08 | [CUDA Graph、形状选择与 padding](chapters/08_cuda_graph.md) | 已编写并核对引用 |
| 09 | [投机解码与验证预算的研究入口](chapters/09_speculative_decoding.md) | 已编写并核对引用 |
| 10 | [正确性、性能分析与最小改进实验](chapters/10_experiments_and_profiling.md) | 协议与脚本已编写；GPU 实验待运行 |

配套材料：[学习路线](roadmap.md)、[增量术语表](glossary.md)、[实际检查记录](checks.md)。第一章附带可复用的环境探测、最小生成示例和源码引用检查脚本，均放在 `tools/`；不放入业务代码或仓库测试套件。

选题材料：[18 个非算子优化项目候选：论文、开源代码与 vLLM 集成边界](research/project_candidates_2026-10-16.md)，附检索记录、外部代码快照、硬件门槛、最小实现和实验要求。这是 2026-10-16 的选题初筛，不代表已经验证算法收益或最终通过 PR 查重。

十章写作已完成。第二章先讲完整普通生成，第三至八章逐层深入其状态、资源和执行，第九章再讲完整投机周期，第十章给出最小实验。不要把每章源文件地图当成逐一通读任务：先沿案例追一条边，确认输入、状态修改和下一消费者，再扩展分支。

第十章新增 `tools/experiment_generate.py`，可在环境就绪后保存实际输入／输出与离线计时，并精确比较 token。`artifacts/` 与 trace 输出已忽略；本轮未生成 GPU 结果。2026-10-16 再次探测确认运行依赖仍缺失，初始环境表不构成当前推理可用性承诺。

目录调整依据：

- 保留十章顺序，但第一章提前区分函数调用、ZMQ 消息、线程队列和 GPU 执行，避免把默认多进程路径画成一个 Python 栈。
- 第三章以当前 `vllm/v1/request.py` 的 `Request` 为中心，不预设旧版本的 Sequence/SequenceGroup 是 V1 主对象。
- 第四章明确 `max_num_batched_tokens` 与 `max_num_scheduled_tokens`，以本版 Scheduler 实现为准。
- 第六章主要读 `vllm/v1/worker/gpu/model_runner.py`，并列出 `gpu_model_runner.py` 回退分支；Model Runner V2 不等于另一个引擎架构版本。
- 第九章把“验证预算分配”作为待验证的研究问题；先调查已有实现和文献，再判断是否有改进空间，不提前宣称创新。

主线限定为单 GPU、纯文本、自回归生成、一个 completion。先用显式同步调度与 eager 执行理解因果，再恢复默认异步调度和 CUDA Graph；多模态、MoE、其他硬件、分布式部署只在主线明确后扩展。

## 事实和实验标记

- **源码事实**：来自固定 commit，附仓库相对路径、符号名与 1 起算行号；本地 Markdown 查看器未必能跳转 `#L`，可用 `rg -n` 定位。
- **设计解释**：对代码职责和取舍的教学解释，不等于作者原话或性能证明。
- **推断／待核实**：尚缺运行或其他代码证据，明确说明验证办法。
- **教学示例**：人为构造的 token ID、块号和计数，绝非日志。
- **已运行**与**仅提供步骤**：分开记录；命令能解析、测试能收集、推理能完成、性能有收益是四种不同证据。

## 开发约束

已阅读根目录 [AGENTS.md](../AGENTS.md)、[CONTRIBUTING.md](../CONTRIBUTING.md) 和 [开发指南](../docs/contributing/README.md)，并参考安装说明、架构概览、Model Runner V2 设计说明与 profiling 指南。当前范围不涉及 Rust 下的额外 AGENTS.md，也不修改代理规则或领域指南。

Python 命令统一通过现有 uv 和 `.venv/bin/python`；本套命令使用 `uv run --no-project --offline --python .venv/bin/python python ...`，不请求依赖变更。仓库开发指南部分示例使用裸 `python`/`pytest`，这里按根 AGENTS.md 改写调用方式。

如果未来贡献代码：先遵守重复工作检查和实质性要求，再设计最小行为测试；影响模型输出或 serving 时执行适当评测。提交者须逐行理解和审阅，PR 披露 AI 协助、查重依据与真实测试结果。本轮仅本地学习材料，不提出 PR，未执行远端 issue/PR 查重。构建 C++/CUDA 前另读 `docs/contributing/incremental_build.md`；本轮没有进行此类修改。
