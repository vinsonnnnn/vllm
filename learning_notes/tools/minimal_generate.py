# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate one completion using an existing local model and environment."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    model_path = args.model.resolve()
    if not model_path.is_dir():
        parser.error("--model must be an existing local model directory")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=1,
        distributed_executor_backend="uni",
        async_scheduling=False,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.3,
        seed=0,
    )
    config = llm.llm_engine.vllm_config
    print("frontend engine:", type(llm.llm_engine).__module__)
    print("core client:", type(llm.llm_engine.engine_core).__name__)
    print("executor config:", config.parallel_config.distributed_executor_backend)
    print("MRV2 selected by config:", config.use_v2_model_runner)
    print("async scheduling:", config.scheduler_config.async_scheduling)
    outputs = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0, max_tokens=2),
        use_tqdm=False,
    )
    for output in outputs:
        print("request_id:", output.request_id)
        print("finished:", output.finished)
        print("prompt_token_ids:", output.prompt_token_ids)
        for completion in output.outputs:
            print("token_ids:", completion.token_ids)
            print("text:", repr(completion.text))
            print("finish_reason:", completion.finish_reason)


if __name__ == "__main__":
    main()
