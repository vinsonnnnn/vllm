# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the pinned source, local links, excerpts and key symbols."""

import ast
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "learning_notes"
PIN = "f9c98bd1ebbddb9acedb1e6d7505ad9588f9a266"

# Selected semantic anchors supplement the line-bound and exact-excerpt checks.
ANCHORS = {
    ("examples/basic/offline_inference/basic.py", 17): "main",
    ("vllm/entrypoints/llm.py", 418): "generate",
    ("vllm/entrypoints/offline_utils.py", 334): "_run_completion",
    ("vllm/v1/engine/llm_engine.py", 223): "add_request",
    ("vllm/v1/engine/llm_engine.py", 304): "step",
    ("vllm/v1/engine/core_client.py", 91): "make_client",
    ("vllm/v1/engine/core_client.py", 976): "add_request",
    ("vllm/v1/engine/core.py", 608): "step",
    ("vllm/v1/engine/core.py", 1422): "run_busy_loop",
    ("vllm/v1/engine/core.py", 1705): "process_input_sockets",
    ("vllm/v1/core/sched/scheduler.py", 535): "schedule",
    ("vllm/v1/core/sched/scheduler.py", 1837): "update_from_output",
    ("vllm/v1/worker/gpu/model_runner.py", 1563): "execute_model",
    ("vllm/v1/worker/gpu/model_runner.py", 1896): "sample_tokens",
    ("vllm/v1/worker/gpu_model_runner.py", 4274): "execute_model",
    ("vllm/config/vllm.py", 653): "use_v2_model_runner",
    ("vllm/renderers/base.py", 582): "_tokenize_prompt",
    ("vllm/renderers/base.py", 1105): "render_cmpl",
    ("vllm/v1/engine/input_processor.py", 281): "process_inputs",
    ("vllm/v1/engine/output_processor.py", 132): "RequestState",
    ("vllm/v1/engine/output_processor.py", 283): "make_request_output",
    ("vllm/v1/engine/output_processor.py", 622): "process_outputs",
    ("vllm/v1/engine/core.py", 999): "preprocess_add_request",
    ("vllm/v1/engine/core.py", 640): "post_step",
    ("vllm/v1/engine/core.py", 649): "step_with_batch_queue",
    ("vllm/v1/request.py", 59): "Request",
    ("vllm/v1/request.py", 238): "from_engine_core_request",
    ("vllm/v1/request.py", 265): "append_output_token_ids",
    ("vllm/v1/request.py", 364): "RequestStatus",
    ("vllm/v1/core/sched/utils.py", 94): "check_stop",
    ("vllm/v1/core/sched/output.py", 219): "SchedulerOutput",
    ("vllm/v1/outputs.py", 320): "ModelRunnerOutput",
    ("vllm/v1/core/sched/scheduler.py", 1440): "_preempt_request",
    ("vllm/v1/core/sched/scheduler.py", 1483): "_update_after_schedule",
    ("vllm/v1/core/sched/scheduler.py", 2305): "_update_request_with_output",
    ("vllm/v1/core/sched/scheduler.py", 2515): "_free_request",
    ("vllm/v1/core/sched/async_scheduler.py", 19): "_update_after_schedule",
    ("vllm/v1/core/sched/async_scheduler.py", 51): "_update_request_with_output",
    ("vllm/v1/core/kv_cache_manager.py", 228): "get_computed_blocks",
    ("vllm/v1/core/kv_cache_manager.py", 343): "allocate_slots",
    ("vllm/v1/core/kv_cache_coordinator.py", 66): "KVCacheCoordinator",
    ("vllm/v1/core/kv_cache_utils.py", 622): "hash_block_tokens",
    ("vllm/v1/core/block_pool.py", 713): "touch",
    ("vllm/v1/core/block_pool.py", 734): "free_blocks",
    ("vllm/v1/core/single_type_kv_cache_manager.py", 714): "find_longest_cache_hit",
    ("vllm/v1/core/single_type_kv_cache_manager.py", 425): "_apply_cow",
    ("vllm/v1/core/single_type_kv_cache_manager.py", 823): "_cache_partial_tail_block",
    ("vllm/v1/worker/gpu/block_table.py", 191): "compute_slot_mappings",
    ("vllm/v1/worker/gpu/states.py", 9): "RequestState",
    ("vllm/v1/worker/gpu/model_runner.py", 1055): "add_requests",
    ("vllm/v1/worker/gpu/model_runner.py", 1113): "update_requests",
    ("vllm/v1/worker/gpu/model_runner.py", 1149): "gather_batch_req_state",
    ("vllm/v1/worker/gpu/model_runner.py", 1202): "prepare_inputs",
    ("vllm/v1/worker/gpu/model_runner.py", 1405): "prepare_attn",
    ("vllm/v1/worker/gpu/model_runner.py", 1444): "sample",
    ("vllm/v1/attention/selector.py", 102): "get_attn_backend",
    ("vllm/model_executor/models/opt.py", 114): "forward",
    ("vllm/model_executor/layers/attention/attention.py", 483): "forward",
    (
        "vllm/model_executor/layers/attention/attention.py",
        706,
    ): "unified_kv_cache_update",
    (
        "vllm/model_executor/layers/attention/attention.py",
        750,
    ): "unified_attention_with_output",
    ("vllm/v1/worker/gpu/async_utils.py", 167): "get_output",
    ("vllm/v1/worker/gpu/cudagraph_utils.py", 59): "BatchExecutionDescriptor",
    ("vllm/v1/worker/gpu/cudagraph_utils.py", 98): "_is_compatible",
    ("vllm/v1/worker/gpu/cudagraph_utils.py", 343): "capture",
    ("vllm/v1/worker/gpu/cudagraph_utils.py", 435): "dispatch",
    ("vllm/v1/worker/gpu/cudagraph_utils.py", 468): "run_fullgraph",
    ("vllm/config/compilation.py", 53): "CUDAGraphMode",
    ("vllm/v1/worker/gpu/spec_decode/__init__.py", 8): "init_speculator",
    ("vllm/v1/worker/gpu/spec_decode/dflash/speculator.py", 323): "propose",
    ("vllm/v1/worker/gpu/spec_decode/rejection_sampler.py", 260): "__call__",
    (
        "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py",
        484,
    ): "_rejection_kernel",
    ("vllm/v1/spec_decode/dynamic/utils.py", 77): "build_dynamic_sd_schedule_lookup",
    ("vllm/v1/worker/gpu/spec_decode/adaptive_verification.py", 269): "get_num_tokens",
    (
        "vllm/v1/worker/gpu/spec_decode/adaptive_verification.py",
        379,
    ): "reallocate_drafts",
    (
        "vllm/entrypoints/cli/benchmark/throughput.py",
        10,
    ): "BenchmarkThroughputSubcommand",
    ("vllm/benchmarks/throughput.py", 37): "run_vllm",
    ("vllm/benchmarks/serve.py", 563): "calculate_metrics",
    ("vllm/config/profiler.py", 39): "ProfilerConfig",
    ("examples/features/profiling/simple_profiling_offline.py", 19): "main",
}


def main():
    git_env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    git = shutil.which("git.exe") or "git"
    head = subprocess.check_output(
        [git, "rev-parse", "HEAD"], cwd=ROOT, text=True, env=git_env
    ).strip()
    ancestor = subprocess.run(
        [git, "merge-base", "--is-ancestor", PIN, "HEAD"], cwd=ROOT, env=git_env
    )
    if ancestor.returncode != 0:
        raise SystemExit(f"HEAD {head} does not descend from source baseline {PIN}")
    unchanged = subprocess.run(
        [
            git,
            "-c",
            "core.autocrlf=true",
            "diff",
            "--quiet",
            PIN,
            "--",
            ".",
            ":(exclude)learning_notes/**",
            ":(exclude)优化/vllm_dflash_dcut_plan_and_prompts.md",
        ],
        cwd=ROOT,
        env=git_env,
    )
    if unchanged.returncode != 0:
        raise SystemExit("Tracked files outside the allowed notes differ from baseline")

    links = citations = excerpts = chapters = 0
    for note in sorted(NOTES.rglob("*.md")):
        content = note.read_text(encoding="utf-8")
        if note.parent.name == "chapters":
            sections = re.findall(r"^## (\d+)\.", content, re.M)
            if sections != [str(i) for i in range(1, 11)]:
                raise SystemExit(f"Missing chapter sections: {note.name}")
            goals = content.split("## 2.", 1)[0]
            questions = content.split("## 9.", 1)[1].split("## 10.", 1)[0]
            answers = content.split("### 自我检验答案", 1)[-1]
            goal_count = len(re.findall(r"^\d+\. ", goals, re.M))
            question_count = len(re.findall(r"^\d+\. ", questions, re.M))
            answer_count = len(re.findall(r"^\d+\. ", answers, re.M))
            if not 3 <= goal_count <= 5 or not 5 <= question_count <= 8:
                raise SystemExit(f"Chapter question counts differ: {note.name}")
            if answer_count != question_count or "### 自我检验答案" not in content:
                raise SystemExit(f"Missing chapter answers: {note.name}")
            chapters += 1
        for label, href in re.findall(r"\[([^\]\n]+)\]\(([^)\n]+)\)", content):
            label = label.replace(r"\_", "_")
            if "://" in href or href.startswith("#"):
                continue
            target, _, fragment = href.partition("#")
            path = (note.parent / unquote(target)).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise SystemExit(f"Invalid local link in {note.name}: {href}")
            links += 1
            match = re.fullmatch(r"(.+):(\d+)", label)
            if match:
                source, line = match[1], int(match[2])
                if path != (ROOT / source).resolve():
                    raise SystemExit(f"Citation target differs from label: {label}")
                if fragment != f"L{line}":
                    raise SystemExit(f"Citation fragment differs from label: {label}")
                if not 1 <= line <= len(path.read_text(encoding="utf-8").splitlines()):
                    raise SystemExit(f"Source line out of range: {label}")
                citations += 1

        pattern = (
            r"<!-- source: (\S+):(\d+)-(\d+) -->\s*"
            r"```python\n(.*?)\n```"
        )
        for source, start, end, quoted in re.findall(pattern, content, re.S):
            original = (ROOT / source).read_text(encoding="utf-8").splitlines()
            selected = "\n".join(original[int(start) - 1 : int(end)])
            if textwrap.dedent(selected) != quoted:
                raise SystemExit(f"Excerpt differs: {source}:{start}-{end}")
            excerpts += 1

    for (source, line), name in ANCHORS.items():
        tree = ast.parse((ROOT / source).read_text(encoding="utf-8"))
        if not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.lineno == line
            and node.name == name
            for node in ast.walk(tree)
        ):
            raise SystemExit(f"Symbol anchor differs: {source}:{line} {name}")

    scripts = list((NOTES / "tools").glob("*.py"))
    for script in scripts:
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    for path in sorted(NOTES.rglob("*")):
        if not path.is_file() or path.suffix not in (".md", ".py"):
            continue
        content = path.read_text(encoding="utf-8")
        if not content.endswith("\n"):
            raise SystemExit(f"Missing final newline: {path}")
        for line_number, line in enumerate(content.splitlines(), 1):
            if line != line.rstrip():
                raise SystemExit(f"Trailing whitespace: {path}:{line_number}")
            if path.suffix == ".py" and len(line) > 88:
                raise SystemExit(f"Python line exceeds 88: {path}:{line_number}")
    print(f"Source baseline OK: {PIN}; current HEAD: {head}")
    print(f"Local links OK: {links}; source-line citations OK: {citations}")
    print(f"Exact excerpts OK: {excerpts}; symbol anchors OK: {len(ANCHORS)}")
    print(f"Chapter sections, goals, questions and answers OK: {chapters}")
    print(f"Python syntax OK: {len(scripts)} scripts; no inference performed")
    print("Note whitespace and Python line lengths OK")


if __name__ == "__main__":
    main()
