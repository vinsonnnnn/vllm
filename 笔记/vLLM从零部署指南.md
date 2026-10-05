# vLLM 从零部署指南（Windows + WSL2 + NVIDIA GPU）

本文说明如何在 Windows 电脑上通过 WSL2 Ubuntu 安装当前 vLLM 仓库，并启动一个可调用的模型推理 API。

## 1. 环境与部署路线

本次检查到的环境：

| 项目 | 信息 |
| --- | --- |
| Windows 原始项目目录 | `E:\vllm` |
| 推荐的 WSL 运行目录 | `~/projects/vllm` |
| 宿主系统 | Windows |
| 显卡 | NVIDIA GeForce RTX 4090 Laptop GPU |
| 显存 | 约 16GB |
| Python 环境 | 使用 `uv` 管理，选择 Python 3.12 |
| 入门模型 | `Qwen/Qwen2.5-1.5B-Instruct` |

部署路线：Windows → WSL2 Ubuntu → 安装当前仓库 → 下载模型 → 启动 API → 发送请求验证。

官方 vLLM 不原生支持 Windows。这里使用 WSL2 提供 Linux 运行环境，使用预编译组件安装当前源码，先跑通小模型。

这份文档是操作指南，并不表示服务已经部署或验证成功。

## 2. 安装或进入 WSL2 Ubuntu

本节命令在 **Windows PowerShell** 中执行，安装操作使用管理员终端。

先检查已安装的 Linux 发行版：

```powershell
wsl --list --verbose
```

如果已有 Ubuntu，且 `VERSION` 为 `2`，直接进入：

```powershell
wsl -d Ubuntu
```

如果没有 Ubuntu，执行安装：

```powershell
wsl --install -d Ubuntu
```

按提示重启电脑，然后打开 Ubuntu，设置 Linux 用户名和密码。输入密码时终端不显示字符，这是正常现象。

如果已有 Ubuntu 使用的是 WSL1，在 PowerShell 中转换为 WSL2：

```powershell
wsl --set-version Ubuntu 2
```

命令中的 `Ubuntu` 需要与 `wsl --list --verbose` 显示的发行版名称一致。

## 3. 在 Ubuntu 中确认 GPU 可用

**以下步骤的命令全部在 Ubuntu 终端中执行，不要直接粘贴到 PowerShell。**

```bash
nvidia-smi
```

应该能看到 RTX 4090 Laptop GPU 及显存信息。如果命令失败，先解决 WSL 的 GPU 访问问题，再继续安装项目。

## 4. 安装工具并创建 Python 环境

Windows 和 Ubuntu 的工具环境相互独立。即使 Windows 已安装 `uv`，也需要在 Ubuntu 中确认它可用。

安装基础工具：

```bash
sudo apt update
sudo apt install -y curl git build-essential rsync
```

如果 Ubuntu 尚未安装 `uv`，执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

将当前仓库复制到 WSL 的 Linux 文件系统，保留源码、Git 历史和本地修改，排除原来的虚拟环境：

```bash
mkdir -p "$HOME/projects/vllm"
rsync -a --exclude='.venv/' /mnt/e/vllm/ "$HOME/projects/vllm/"
cd "$HOME/projects/vllm"
```

WSL 中的 `/mnt/e/vllm` 对应 Windows 中的 `E:\vllm`。在 `/mnt/e` 下执行 Linux 构建会增加跨文件系统访问开销，本次实际安装出现了 `git status` 超过 40 秒的报错，因此推荐从 Linux 目录运行。复制可能需要一些时间，完成后再继续。

Windows 原始目录仍保留。后续修改运行代码时，请编辑 `~/projects/vllm` 中的副本，两份目录不会自动同步。可以在 Ubuntu 中执行 `explorer.exe .` 打开当前目录。

首次使用时创建虚拟环境：

```bash
uv venv --python 3.12
source .venv/bin/activate
```

如果已经在新的 Linux 项目目录中创建了 `.venv`，只需执行激活命令，无需重复创建。迁移时不要复制旧 `.venv`，因为它可能包含旧目录的绝对路径。Windows 创建的虚拟环境也不能直接用于 Linux。

遵循仓库要求：使用 `uv` 管理依赖，Python 命令使用 `.venv/bin/python`，不要使用系统 `python3` 或裸 `pip`。

## 5. 安装当前 vLLM 项目

先安装仓库要求的检查工具和 Git hooks：

```bash
uv pip install -r requirements/lint.txt
pre-commit install
```

使用预编译组件进行可编辑安装：

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

参数含义：

- `VLLM_USE_PRECOMPILED=1`：下载预编译组件，避免首次安装时完整编译底层代码。
- `-e .`：安装当前仓库，修改 Python 源码后重启服务即可使用修改后的代码。
- `--torch-backend=auto`：由 `uv` 根据驱动选择合适的 PyTorch 后端。

这种安装方式适合先运行项目和修改 Python 代码。如果修改 C++ 或 CUDA 代码，需要按照仓库的增量编译指南重新构建。

安装完成后检查：

```bash
.venv/bin/python -c "import torch, vllm; print('vLLM:', vllm.__version__); print('CUDA:', torch.cuda.is_available())"
```

预期显示 vLLM 版本号和 `CUDA: True`。若导入失败或 CUDA 不可用，先解决安装问题，再启动服务。

## 6. 启动模型推理服务

先使用仓库快速入门中的小模型跑通流程：

当前源码的 V2 Model Runner 需要 UVA，而 WSL 下默认关闭了它依赖的锁页内存。本次启动因此出现 `RuntimeError: UVA is not available`，这里显式切换到 V1 Model Runner，先绕过这个初始化错误。

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7
```

| 参数 | 作用 |
| --- | --- |
| `VLLM_USE_V2_MODEL_RUNNER=0` | 为这次启动选择 V1 Model Runner，绕过当前 WSL 环境下 V2 runner 的 UVA 依赖 |
| 模型名称 | 指定要加载的 Hugging Face 模型 |
| `--host 127.0.0.1` | 监听本机地址 |
| `--port 8000` | 使用 8000 端口 |
| `--dtype float16` | 使用半精度模型权重 |
| `--max-model-len 4096` | 设置单次请求的上下文长度上限，包含输入和输出 |
| `--gpu-memory-utilization 0.7` | 设置 GPU 显存使用比例目标，为桌面和其他程序留出空间 |

首次启动会下载模型，需要能够访问 Hugging Face，并预留足够的磁盘空间。加载、编译和初始化需要时间。

看到 `Application startup complete` 等服务就绪信息后，保持这个终端运行。

vLLM 提供模型推理 API；如需聊天网页，需要另外连接前端。

## 7. 发送请求验证

另外打开一个 Ubuntu 终端。下面的 `curl` 请求不需要激活虚拟环境。

查看已加载模型：

```bash
curl http://127.0.0.1:8000/v1/models
```

测试对话：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "请用中文简单介绍你自己"}
    ],
    "max_tokens": 128
  }'
```

返回 JSON，且 `choices` 中有模型回答，就说明服务和推理调用已经跑通。

这里的 `model` 必须与启动命令中的模型名称一致。

### 7.1 如何确认已经启动成功

本次用户截图中已出现：

- `Application startup complete`：API 服务完成启动。
- `GET /v1/models ... 200 OK`：模型查询成功。
- `POST /v1/chat/completions ... 200 OK`，且返回模型回答：对话推理成功。

这些信息说明本次部署和实际调用均已成功。没有请求时，`Running: 0`、`Waiting: 0` 和吞吐量为 0 是空闲状态。

### 7.2 在终端中连续聊天

保持运行服务的终端打开，在另一个 Ubuntu 终端执行：

```bash
cd "$HOME/projects/vllm"
source .venv/bin/activate

vllm chat \
  --url http://127.0.0.1:8000/v1 \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --api-key EMPTY
```

出现 `>` 后输入问题并按回车，可以连续对话，回答会逐步显示。该客户端会在当前会话中保留历史；重新启动客户端会开始新会话。对话过长会超过当前 4096 token 的上下文限制，此时应开始新会话。

按 `Ctrl+D` 退出聊天客户端，服务终端继续运行。`EMPTY` 是当前未启用 API key 验证时的客户端占位值。

只问一次可以使用：

```bash
vllm chat --url http://127.0.0.1:8000/v1 \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --api-key EMPTY \
  --quick "用中文解释什么是大语言模型"
```

### 7.3 在 Python 程序中调用

当前项目依赖包含 `openai` 客户端包，可以通过兼容接口调用本地 vLLM 服务。下面的请求发往 `127.0.0.1`，使用本地加载的 Qwen 模型。

保存为 Linux 项目目录中的 `call_model.py`：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "user", "content": "请用中文介绍一下 Python。"},
    ],
    max_tokens=256,
)

print(response.choices[0].message.content)
```

在该目录执行：

```bash
.venv/bin/python call_model.py
```

如果要实现多轮聊天，需要将之前的用户消息和模型回答一并放入后续请求的 `messages`。单独的新请求不会自动继承之前的聊天历史。

## 8. 后续启动与停止

以后无需重新安装，在 Ubuntu 中执行：

```bash
cd "$HOME/projects/vllm"
source .venv/bin/activate

VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7
```

在运行服务的终端按 `Ctrl+C` 停止服务。

## 9. 遇到问题时先检查什么

| 现象 | 检查方向 |
| --- | --- |
| `git status` 超过 40 秒，安装失败 | 按第 4 节复制仓库到 Linux 文件系统，在新目录创建虚拟环境并重新安装 |
| 验证时出现 `No module named 'torch'` | 先检查前面的项目安装是否成功；本次截图中这是安装失败后的后续报错 |
| `RuntimeError: UVA is not available` | 按第 6 节在启动命令前加 `VLLM_USE_V2_MODEL_RUNNER=0`，切换到 V1 runner |
| WSL 返回“拒绝访问” | 在本机管理员 PowerShell 中检查 WSL 状态和安装情况 |
| Ubuntu 中 `nvidia-smi` 失败 | 检查是否为 WSL2，以及 Windows NVIDIA 驱动和 WSL GPU 支持 |
| 找不到 `uv` | 确认在 Ubuntu 中安装，并执行 `source "$HOME/.local/bin/env"` |
| 找不到 `vllm` | 确认安装成功，并执行 `source .venv/bin/activate` |
| 显示 `CUDA: False` | 检查 WSL GPU 是否可见，以及 PyTorch 安装是否正确 |
| 模型下载失败 | 检查 Ubuntu 的网络连接和 Hugging Face 访问情况 |
| 停在 `Using FlashAttention version 2` | 这行是正常信息；检查模型权重下载或加载进度，参见第 9.3 节 |
| 显存不足 | 关闭其他占用 GPU 的程序，先使用小模型和较短上下文，并查看报错中的显存需求 |
| 8000 端口被占用 | 更换启动命令的端口，并同步修改请求地址 |
| 请求连接失败 | 确认服务已经就绪且运行终端没有退出，先在 Ubuntu 内测试 |
| 客户端报 `UnicodeEncodeError ... surrogates not allowed` | 请求文本含有不能编码的代理码点，先按第 9.4 节检查终端输入编码 |

如果需要进一步定位，记录执行的命令和最后一段报错。

### 9.1 本次安装超时的修复步骤

截图中的首个错误是：

```text
subprocess.TimeoutExpired: Command '['git', '--git-dir', '/mnt/e/vllm/.git', 'status', '--porcelain', '--untracked-files=no']' timed out after 40 seconds
```

仓库构建使用 `setuptools-scm` 获取版本信息，需要查询 Git 状态。这一步超时导致安装失败。随后当前虚拟环境显示 `No module named 'torch'`，不能据此判断 GPU 或驱动有问题，也不应只单独安装一个任意版本的 PyTorch 来处理。

跨文件系统访问是最可能的原因，尚未在迁移后的目录验证。Microsoft 建议 Linux 工具使用 WSL 的 Linux 文件系统，以获得更好的性能。

对于截图中已经激活旧环境的终端，按以下顺序执行：

```bash
deactivate

sudo apt install -y rsync
mkdir -p "$HOME/projects/vllm"
rsync -a --exclude='.venv/' /mnt/e/vllm/ "$HOME/projects/vllm/"
cd "$HOME/projects/vllm"

time git status --porcelain --untracked-files=no

uv venv --python 3.12
source .venv/bin/activate

uv pip install -r requirements/lint.txt
pre-commit install
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

如果终端没有激活虚拟环境，跳过 `deactivate`。如果 Git 状态查询仍然长时间不返回，先检查新目录的位置和 Git 问题，再继续安装。

只有安装成功后才执行：

```bash
.venv/bin/python -c "import torch, vllm; print('vLLM:', vllm.__version__); print('CUDA:', torch.cuda.is_available()); print('源码:', vllm.__file__)"
```

确认 `CUDA: True`，源码路径指向新的 Linux 项目目录，再按第 6 节启动服务。

### 9.2 WSL 下 V2 Model Runner 初始化失败

本次日志中的关键行：

```text
Using V2 Model Runner
RuntimeError: UVA is not available
```

报错发生在 `RequestState` 创建 `UvaBuffer` 时。当前代码要求 UVA 可用，而 UVA 检查依赖锁页内存检查。CUDA 平台在 WSL 下默认禁用锁页内存，导致该初始化失败。这段日志没有显示显存不足或模型权重下载失败。

先按第 6 节使用 `VLLM_USE_V2_MODEL_RUNNER=0` 启动，无需为这个错误重新安装依赖。该变量只作用于紧随其后的启动命令，后续每次启动都应保留。

这项处理是根据当前仓库配置分支确认的绕过方法，尚未在用户的 WSL 环境中验证服务启动成功。启动就绪后，按第 7 节发送请求验证。

当前 CUDA 实现还提供 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`，用于在符合内核要求的 WSL2 上启用锁页内存；它需要验证实际环境支持情况，本指南先采用 V1 runner 方案。

定位依据：

- `vllm/config/vllm.py`：显式设置 `VLLM_USE_V2_MODEL_RUNNER=0` 时选择 V1 runner。
- `vllm/v1/worker/gpu/buffer_utils.py`：UVA 检查失败时抛出本次错误。
- `vllm/utils/platform_utils.py`：GPU 上的 UVA 可用性依赖锁页内存检查。
- `vllm/platforms/cuda.py`：WSL 下的锁页内存默认关闭。

### 9.3 停在 FlashAttention 日志之后

如果最后几行是：

```text
Starting to load model Qwen/Qwen2.5-1.5B-Instruct...
Using FlashAttention version 2
```

它们都是正常日志，只能说明进入了模型加载阶段，不能单凭截图认定进程卡死或正在编译。当前代码在构造模型后准备并加载权重，权重下载使用了禁用进度条的 `DisabledTqdm`，首次下载时可能长时间没有新输出。

先保留服务终端，在另一个 Ubuntu 终端查看默认缓存目录的大小变化：

```bash
watch -n 2 'du -sb "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"'
```

数字持续增大说明缓存文件仍在写入，可以继续等待。数字不变不能单独证明进程卡死，还可能是在请求网络元数据、等待下载锁、初始化模型或读取已下载文件。如果设置了 `HF_HOME`、`HF_HUB_CACHE` 或 `--download-dir`，应查看实际缓存目录。

如果持续数分钟没有新日志，且看不到下载进展，可以在服务终端按 `Ctrl+C` 停止，然后在已激活虚拟环境的终端显式下载模型，查看进度或网络错误：

```bash
cd "$HOME/projects/vllm"
source .venv/bin/activate
hf download Qwen/Qwen2.5-1.5B-Instruct
```

`hf download` 会使用 Hugging Face 缓存，复用已下载的文件。下载成功后按第 6 节重新启动。如果下载也长时间无进展或报错，保留完整输出以定位网络问题；模型主页可访问不代表大文件下载链路一定畅通。

### 9.4 聊天客户端出现字符编码错误

如果出现：

```text
UnicodeEncodeError: 'utf-8' codec can't encode characters ... surrogates not allowed
```

本次堆栈显示错误发生在客户端把 JSON 请求编码为 UTF-8 时，请求尚未发出。输入文本中含有不能直接编码为 UTF-8 的代理码点；终端输入编码不一致或粘贴内容异常是可能原因，不能仅凭截图确定具体来源。

截图最下面的 `Please enter a message for the chat model:` 和 `>` 表示新启动的聊天客户端正在等待输入。先在 `>` 后输入纯英文 `Hello` 并回车，确认这次能否完成请求。`argument 'url' is deprecated` 是警告，截图中客户端仍进入了聊天输入状态。

如果英文正常而中文仍报编码错误，在聊天客户端按 `Ctrl+D` 退出，保持服务终端运行，再以 UTF-8 环境重新启动客户端：

```bash
cd "$HOME/projects/vllm"
source .venv/bin/activate

LC_ALL=C.UTF-8 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 vllm chat \
  --url http://127.0.0.1:8000/v1 \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --api-key EMPTY
```

出现 `>` 后先手动输入简单中文，例如“你好”，不要粘贴之前失败的文本。这些环境变量只作用于该客户端进程，是否解决本次问题仍需实际验证。

如果仍失败，退出客户端并检查输入编码以及实际读取的字符：

```bash
.venv/bin/python -c "import sys, locale; print('stdin:', sys.stdin.encoding, sys.stdin.errors); print('locale:', locale.getencoding()); print(ascii(input('Input: ')))"
```

在 `Input:` 后输入同样的文字。`ascii()` 会把非 ASCII 字符转成转义形式，便于区分正常中文码点和 `\udcXX` 等代理码点。保留输出和完整错误，进一步判断是终端、输入文本还是环境变量问题。

## 10. 参考资料

- [vLLM 官方 GPU 安装指南](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [Microsoft WSL 安装指南](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Microsoft WSL GPU 计算说明](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
- [Microsoft 跨文件系统工作与性能建议](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
- [vLLM 环境变量说明](https://docs.vllm.ai/en/stable/configuration/env_vars/)
- [vLLM 终端聊天命令说明](https://docs.vllm.ai/en/stable/cli/chat/)
- [Hugging Face 下载文件说明](https://huggingface.co/docs/huggingface_hub/main/guides/download)
- [Python UTF-8 模式与标准输入编码](https://docs.python.org/3.12/using/cmdline.html)
- 仓库快速入门：`docs/getting_started/quickstart.md`
- 仓库预编译安装说明：`docs/getting_started/installation/gpu.cuda.inc.md`
- 仓库增量编译指南：`docs/contributing/incremental_build.md`
