# 核实与检查记录

日期：2026-10-16（Asia/Shanghai）。源码 commit 固定为 `f9c98bd1ebbddb9acedb1e6d7505ad9588f9a266`。本文件记录实际发生的检查，不把计划步骤写成运行结果。

## 仓库与说明文件

- `git branch --show-current` → `main`；`git rev-parse HEAD` → 上述完整 commit。
- `git log -1 --format=fuller` → `docs: add personal vLLM deployment guide`，提交时间 2026-10-05 11:17:01 +0800。
- `git describe --tags --always --dirty` → `f9c98bd1e`；`git tag --points-at HEAD` 无输出；`git remote -v` 确认 fork origin。
- 开始时 `git status --porcelain=v1` 无输出。既有 `笔记/` 保留，本轮只新增 `learning_notes/`。
- `rg --files --hidden -g AGENTS.md` 找到根目录、`rust/` 与 `rust/src/bench/` 的说明；学习笔记适用根目录规则。
- 已读根 AGENTS.md、CONTRIBUTING.md、`docs/contributing/README.md`；参考当前树内的 `docs/getting_started/installation/gpu.md`、CPU 安装说明、`docs/design/arch_overview.md`、`docs/design/model_runner_v2.md`、`docs/contributing/profiling.md` 和测试 conftest。架构文档仅作导读，路径分支以源码核对结果为准。
- `pyproject.toml` 声明动态版本；`vllm/_version.py` 不存在；不导入 vLLM 猜测发行版本。

## 环境探测

Windows PowerShell 的 PATH 探测发现 uv、nvidia-smi 与 wsl；没有找到 pytest、pre-commit、nvcc、nsys、ncu 的 Windows 命令。`Get-CimInstance` 操作系统与显卡查询被拒绝，后来用 .NET OSVersion 和 nvidia-smi 获取必要信息；未把拒绝访问误写成硬件缺失。

沙箱内直接调用 WSL 最初失败；使用获准的只读环境检查后，确认默认 Ubuntu 是 WSL2，并能进入 Linux shell。首次 `/bin/sh` 中找不到 uv；改用 `/bin/bash -lc` 找到 `/home/vinson/.local/bin/uv`。这属于命令环境差异，不是新安装了 uv。

Windows 中的 uv 尝试因用户缓存目录访问被拒绝，未据此认定 Python 不存在；随后在 WSL 验证了现有解释器符号链接和 Python 3.12.15。没有使用系统 Python 或 bare pip。

实际执行环境探测命令（初次额外带 `--no-sync`，uv 提示与 `--no-project` 同用无效；笔记最终命令去掉这个冗余选项）：

```bash
cd /mnt/e/vllm
uv run --no-project --offline --python .venv/bin/python python learning_notes/tools/probe_environment.py
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nsys --version
```

探测结果：Linux 内核 6.18.33.2、Python 3.12.15、WSL uv 0.12.23；pre-commit 4.6.2；vLLM/Torch/Triton/Transformers/pytest 未安装。GPU 名称、驱动和显存见首页。Nsight Systems 版本可读取，`nvcc` 和 `ncu` 不在 WSL PATH。没有 Torch，因此未执行 `torch.cuda.is_available()`，不报告 True/False。

## CPU 测试与推理

已尝试：

```bash
uv run --no-project --offline --python .venv/bin/python python -m pytest tests/v1/core/test_scheduler.py --collect-only -q
```

结果为 `/mnt/e/vllm/.venv/bin/python: No module named pytest`，未收集和执行任何测试。批量检查命令最后执行 `nsys --version` 成功，不代表中途 pytest 成功；这里按子命令实际输出分别记录。

GPU 推理、模型评测、benchmark、Nsight trace 采集均未运行。模型快照可用性尚未核实；最小脚本要求用户以后提供已有本地模型目录，不在本轮下载或安装。

## 第一章源码核对

人工按源码核对入口别名、offline mixin、前端请求注册、client 选择、ZMQ ADD、核心输入线程预处理与队列、核心循环、调度执行更新、uni 的本进程 Worker、两套 Runner 选择、MRV2 eager 前向和前端最终输出收集。重点排除了以下误读：

1. `vllm/engine/` 文件存在不说明运行 V0。
2. 单 GPU 与 `uni` 不说明前端和核心一定同进程。
3. `collective_rpc` 或 Future 不自动意味着跨进程或后台线程。
4. 前端 step 与核心 step 不是同一职责。
5. MRV1 与 MRV2 不能拼成一条执行链。
6. `num_computed_tokens` 存在调度预记账，教学表不能当作字段更新时序日志。

自动检查首次发现原示例 `main` 的引用偏移一行，已修正为第 17 行并重新运行。最终检查命令：

```bash
uv run --no-project --offline --python .venv/bin/python python learning_notes/tools/check_sources.py
uv run --no-project --offline --python .venv/bin/python python learning_notes/tools/minimal_generate.py --help
uv run --no-project --offline --python .venv/bin/python python -m pre_commit validate-config
```

结果：固定 HEAD 匹配；63 个本地链接目标存在；52 个源码行号引用有效；5 段源码摘录与固定源码去除外围缩进后逐字一致；16 个关键符号通过 AST 行号与名称核对；3 个附带 Python 脚本语法通过。最小脚本的 `--help` 成功，仅验证命令行解析，不导入 vLLM、不执行生成。pre-commit 配置验证成功，这不等于运行了全部 hooks；没有为格式检查下载依赖。

首轮收尾 `git diff --check` 无输出，`git diff --name-only` 无输出，`git status --short` 只显示 `?? learning_notes/`。前两条只检查已跟踪文件，不能单独证明新增 Markdown 格式通过；新增文件由源码检查脚本和人工阅读核对，另检查空白与行尾。首轮未运行 Markdownlint、完整 pre-commit hooks、pytest 测试函数或 GPU 推理。

## 用户授权自动提交与推送后的检查

用户随后明确要求每轮更改后自动推送。已将授权与目标仓库记入首页；源码仍固定在原 commit，允许仅学习笔记的提交推进 HEAD。引用检查脚本改为核对基线祖先关系和笔记目录外的已跟踪文件差异；只读 Git 命令禁用可选索引锁。在此 Windows/WSL 共享工作树中优先调用现有 Windows Git，并统一处理 CRLF，避免两种 Git 的换行配置差异造成误报和重复扫描。

提交准备时运行 pre-commit，尽管存在缓存，它仍自动初始化了部分独立 hook 工具环境。发现后停止该进程；没有安装或升级项目 `.venv` 中的推理依赖，不将这次尝试报告成 hooks 通过。本次提交单独跳过自动 hooks，保留仓库原有 hook 配置；直接使用已有可执行工具检查，不再触发安装。

实际完成：

- 缓存中的 Ruff 对三个辅助脚本执行 `check`，结果通过；`format` 修正一份脚本的排版。脚本补齐仓库 SPDX 头，设备计数采用 `torch.accelerator.device_count()`。
- 缓存中的 Markdownlint CLI 0.21.0 对五份 Markdown 执行检查，修正粗体风格后为 `0 error(s)`。
- 更新后的 `check_sources.py` 通过：63 个本地链接、52 处源码引用、5 段原文、16 个符号锚点、3 个脚本语法以及行尾空白／Python 行长。
- 远端 `refs/heads/main` 在推送前核实为原始源码基线，目标为 `origin` 的 `vinsonnnnn/vllm`；不创建 PR，不强制推送。

推送成功与提交号在聊天中报告，以免为写入本次提交号制造递归提交。本节不会将推送前检查视作推送成功的证据。

## 剩余章节编写与核对（2026-10-16）

用户要求继续完成其他剩余章节并提交 GitHub。本轮先读首页、路线和第一章总结；起始 HEAD 为 `8787e13e232bebc97704fa90b465c5c79e1ddabe`，工作区干净，远端 main 与之相同。相对固定源码基线，仅已有学习笔记和用户新增的 `优化/vllm_dflash_dcut_plan_and_prompts.md`；业务源码、配置与测试未变化。

因此将基线检查的排除项精确增加上述用户文档；没有把它的实施提示词当成本轮执行指令，没有修改该文档、切换版本或开展 D-Cut 业务实现。第一章保留，新增第二至十章，更新目录、路线和增量术语。

### 本轮源码核对重点

- 普通请求：renderer 实际 tokenize、InputProcessor 构造消息、内部 ID、核心输入线程创建 Request、目标 logits／采样、逐 token 停止、前端 stop string 与 FINAL_ONLY。
- 状态与调度：新／缓存请求增量，输出 ID 映射、调度后预记账、in-flight、两类预算、chunked prefill、抢占重算和 Worker 清理。
- KV 与 Runner：前缀 hash、容量检查、引用计数、部分命中／CoW、块表／slot mapping、持久行与 batch 行、positions／seq_lens、Attention 包装和实际 backend 调用边界。
- 异步与图：batch queue、placeholder、D2H 事件、stale／取消、仅特定 connector 条件开启的 deferred free、MRV2 graph 描述与兼容匹配、padding 和 replay。
- 投机：同步候选交接、MRV2 greedy 前缀验证、纠正／bonus、CPU 和 GPU 拒绝回退、DFlash propose、Dynamic SD 与当前限 DSpark 的 Adaptive Verification。
- 实验：本版 throughput 的输入／输出 token 分母、serve 的 TPOT、profile 控制、最小离线预算对照的测量边界。

引用检查中发现并修正了新摘录范围包含末尾空行，以及两个手工方法行号偏移；另修复 Markdown 格式工具将链接标签中的双下划线路径当成强调的情况，使用转义保留真实文件名，并重新核对链接目标。最终以实际源码和 AST 为准。静态检查只保证路径、行号、符号、原文和结构，因果解释另按调用者、返回值与状态更新人工核查；不能据此认定 GPU 运行通过。

### 本轮实际执行

1. 再次在 WSL 用现有 uv／`.venv/bin/python` 运行 `probe_environment.py`。Python 3.12.15 可用；vLLM、Torch、Triton、Transformers、pytest 仍无安装元数据。没有安装／升级依赖或下载模型；本轮没有触发 pre-commit 环境初始化。
2. 运行扩展后的 `check_sources.py`：十章的十个规定部分、3—5 个目标问题、5—8 个自检问题与对应末尾答案均通过；32 段源码摘录逐字匹配，150 处源码行号引用、79 个 AST 符号锚点通过。脚本还检查本地链接、四份工具脚本语法、行尾空白与 Python 88 列。
3. 直接使用已有缓存中的 Ruff 与 Markdownlint，未运行会安装环境的自动 hooks。Ruff 格式和 lint 检查通过；Markdownlint 检查十四份 Markdown 后为 0 error(s)。SPDX 头按仓库脚本检查。
4. 新增 `experiment_generate.py`；实际运行 `--help` 和比较器合成夹具验证。比较器契约是：输入真实运行记录，配置相容且重复输出一致时报告统计；配置／token／停止原因／计数／时长无效时拒绝比较。使用最便宜的标准库子进程验证层，未导入模型。
5. 八项合成检查通过：相同记录接受、跨配置 token 差异拒绝、同配置重复差异拒绝、错误输出计数拒绝、零时长拒绝、dtype 条件差异拒绝、空运行记录拒绝，以及本地模型目录不存在时在导入 vLLM 前拒绝。合成输入只在临时目录存在并自动清理；没有留下可误认为真实性能结果的 JSON。

未执行：pytest 测试函数、模型推理、模型 eval、prefix cache／异步／graph 的 GPU 对照、DFlash 运行、benchmark 和 profiler 采集。第十章脚本的 `run` 路径及第九章模型示例均仅提供步骤。没有为 GPU 性能填写任何数字；三次同实例重复与独立进程重复在实验协议中分别说明。

原始实验输出和 trace 的预期目录 `learning_notes/artifacts/` 已忽略，防止自动推送大体积产物。最终只暂存本轮学习材料，并将实际提交与远端 SHA 在聊天中报告。
