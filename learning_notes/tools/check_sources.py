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
        ],
        cwd=ROOT,
        env=git_env,
    )
    if unchanged.returncode != 0:
        raise SystemExit("Tracked files outside learning_notes differ from baseline")

    links = citations = excerpts = 0
    for note in sorted(NOTES.rglob("*.md")):
        content = note.read_text(encoding="utf-8")
        for label, href in re.findall(r"\[([^\]\n]+)\]\(([^)\n]+)\)", content):
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
    print(f"Python syntax OK: {len(scripts)} scripts; no inference performed")
    print("Note whitespace and Python line lengths OK")


if __name__ == "__main__":
    main()
