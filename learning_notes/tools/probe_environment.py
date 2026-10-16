# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read existing package metadata without installing or downloading models."""

import importlib.metadata
import importlib.util
import platform
import shutil
import sys

print("platform:", platform.platform())
print("python:", sys.version)
print("executable:", sys.executable)
for name in ("vllm", "torch", "triton", "pytest", "pre-commit", "transformers"):
    try:
        print(f"package {name}:", importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        print(f"package {name}: NOT INSTALLED")
for name in ("uv", "nvidia-smi", "nvcc", "nsys", "ncu"):
    print(f"tool {name}:", shutil.which(name))
if importlib.util.find_spec("torch") is not None:
    try:
        import torch

        print("torch CUDA build:", torch.version.cuda)
        print("torch CUDA available:", torch.cuda.is_available())
        print("torch accelerator count:", torch.accelerator.device_count())
    except Exception as exc:
        print("torch import/probe failed:", repr(exc))
