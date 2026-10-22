"""Dependency-free public-release policy audit.

Run from the repository root with:
    python scripts/release_audit.py
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATH = re.compile(
    r"(^|/)(train|test)\.csv$|\.parquet$|\.safetensors$|\.bin$|\.pt$|\.pth$|\.ckpt$|\.zip$|\.7z$|\.rar$|__pycache__|\.deps/|(^|/)(papers|extracted)/|acl-recovery"
)
SENSITIVE = re.compile(
    r"-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}"
)
MAX_BYTES = 1024 * 1024


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        check=True, capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    files = tracked_files()
    failures: list[str] = []
    for relative in files:
        path = ROOT / relative
        if FORBIDDEN_PATH.search(relative):
            failures.append(f"forbidden tracked path: {relative}")
        if path.stat().st_size > MAX_BYTES:
            failures.append(f"tracked file exceeds 1 MiB: {relative}")
        if path.suffix.lower() in {".py", ".pyw"}:
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except SyntaxError as exc:
                failures.append(f"Python syntax error in {relative}: {exc}")
        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                failures.append(f"JSON parse error in {relative}: {exc}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SENSITIVE.search(text):
            failures.append(f"credential-shaped text: {relative}")

    if failures:
        print("release audit failed:", file=sys.stderr)
        print("\n".join(f"- {item}" for item in failures), file=sys.stderr)
        return 1
    print(f"release audit passed: {len(files)} tracked files; max size <= 1 MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
