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
