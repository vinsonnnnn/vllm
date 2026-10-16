# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record a small offline budget experiment, or compare its actual outputs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def compare(paths):
    baseline, candidate = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    for key in (
        "model",
        "model_revision",
        "requests",
        "max_tokens",
        "memory_fraction",
        "source_baseline",
        "model_config_hashes",
        "effective_max_num_seqs",
        "gpu_inventory",
        "platform",
        "versions",
        "dtype",
        "measurement",
    ):
        if baseline[key] != candidate[key]:
            raise SystemExit(f"Experiment conditions differ: {key}")
    for result in (baseline, candidate):
        if not result["runs"]:
            raise SystemExit("No measured runs")
        for measurement in result["runs"]:
            if measurement["elapsed_seconds"] <= 0:
                raise SystemExit("Measurement duration must be positive")
            outputs = measurement["outputs"]
            if len(outputs) != result["requests"]:
                raise SystemExit("Recorded request count differs")
            token_count = sum(len(output["token_ids"]) for output in outputs)
            if token_count != measurement["output_tokens"]:
                raise SystemExit("Recorded output token count differs")
        reference = result["runs"][0]["outputs"]
        if any(run["outputs"] != reference for run in result["runs"]):
            raise SystemExit("Outputs differ across repeats")
    if baseline["runs"][0]["outputs"] != candidate["runs"][0]["outputs"]:
        raise SystemExit("Prompt IDs, output IDs, or finish reasons differ")
    print("Exact prompt/output token IDs and finish reasons match")
    for label, result in zip(("baseline", "candidate"), (baseline, candidate)):
        rates = [r["output_tokens"] / r["elapsed_seconds"] for r in result["runs"]]
        print(label, "output tokens/s per run:", rates)
        print(label, "median output tokens/s:", statistics.median(rates))
    print("Check manifests, load order and noise before attributing a speedup")


def run(args):
    if not args.model.is_dir() or args.output.exists():
        raise SystemExit("Need an existing local model and a new output path")
    if min(args.budget, args.requests, args.repeats, args.max_tokens) <= 0:
        raise SystemExit("Budget, requests, repeats and max tokens must be positive")
    if not 0 < args.memory_fraction <= 1 or args.max_tokens >= 256:
        raise SystemExit("Need 0 < memory fraction <= 1 and max tokens < 256")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("check_sources.py"))],
        check=True,
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    from vllm import LLM, SamplingParams

    templates = [
        "Explain what a KV cache stores.",
        "A request waits for tokens, memory and execution. " * 8,
        "Explain continuous batching and its relationship to latency. " * 4,
        "List the inputs and outputs of a text generation engine.",
    ]
    prompts = [templates[i % len(templates)] for i in range(args.requests)]
    llm = LLM(
        model=str(args.model.resolve()),
        tensor_parallel_size=1,
        distributed_executor_backend="uni",
        async_scheduling=False,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=256,
        max_num_seqs=min(8, args.requests, args.budget),
        max_num_batched_tokens=args.budget,
        gpu_memory_utilization=args.memory_fraction,
        seed=0,
    )
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    llm.generate(prompts, params, use_tqdm=False)
    runs = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        if len(outputs) != args.requests or not all(o.finished for o in outputs):
            raise RuntimeError("Incomplete generation; do not report throughput")
        records = [
            {
                "prompt_token_ids": list(o.prompt_token_ids),
                "token_ids": list(o.outputs[0].token_ids),
                "finish_reason": o.outputs[0].finish_reason,
            }
            for o in outputs
        ]
        runs.append(
            {
                "elapsed_seconds": elapsed,
                "output_tokens": sum(len(r["token_ids"]) for r in records),
                "outputs": records,
            }
        )
    config = llm.llm_engine.vllm_config
    from check_sources import PIN, ROOT

    git = shutil.which("git.exe") or "git"
    head = subprocess.check_output(
        [git, "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    gpu_inventory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    hashes = {}
    for name in ("config.json", "tokenizer_config.json"):
        path = args.model / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {}
    for name in ("vllm", "torch", "triton", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "no installed metadata"
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_baseline": PIN,
        "git_head": head,
        "gpu_inventory": gpu_inventory,
        "model_config_hashes": hashes,
        "model": str(args.model.resolve()),
        "model_revision": args.model_revision,
        "requests": args.requests,
        "max_tokens": args.max_tokens,
        "memory_fraction": args.memory_fraction,
        "budget": args.budget,
        "effective_scheduled_budget": config.scheduler_config.max_num_scheduled_tokens,
        "effective_max_num_seqs": config.scheduler_config.max_num_seqs,
        "platform": platform.platform(),
        "dtype": str(config.model_config.dtype),
        "command": sys.argv,
        "versions": versions,
        "measurement": "offline generate wall time after one warmup; no TTFT/ITL",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print("Saved actual outputs and timings to", args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    runner = subparsers.add_parser("run")
    runner.add_argument("--model", type=Path, required=True)
    runner.add_argument("--model-revision", required=True)
    runner.add_argument("--budget", type=int, default=64)
    runner.add_argument("--requests", type=int, default=8)
    runner.add_argument("--repeats", type=int, default=3)
    runner.add_argument("--max-tokens", type=int, default=16)
    runner.add_argument("--memory-fraction", type=float, default=0.3)
    runner.add_argument("--output", type=Path, required=True)
    comparer = subparsers.add_parser("compare")
    comparer.add_argument("paths", nargs=2, type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        compare(args.paths)


if __name__ == "__main__":
    main()
