#!/usr/bin/env python3
"""Fail-closed private-server check for the compact GPU control.

This script prints a non-secret JSON report and exits non-zero when --strict is
used and a required condition is missing. It does not download data or run a
model. Use gpu_input_manifest.py to acquire/verify the four compact inputs.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import gpu_input_manifest as input_manifest  # noqa: E402

EXPECTED_LABELS = {
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "claude-fable-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
}
REQUIRED_IMPORTS = (
    "numpy", "pandas", "scipy", "sklearn", "pyarrow",
    "torch", "transformers", "safetensors", "accelerate",
)


def run_capture(command: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return result.returncode, (result.stdout + result.stderr).strip()
    except Exception as exc:
        return 127, type(exc).__name__


def gpu_report() -> dict[str, Any]:
    report: dict[str, Any] = {"nvidia_smi": None, "torch": None}
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        code, text = run_capture([
            nvidia, "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ])
        rows = []
        if code == 0:
            for line in text.splitlines():
                values = [part.strip() for part in line.split(",")]
                if len(values) == 5:
                    rows.append({"index": values[0], "name": values[1], "memory_total_mb": values[2], "memory_free_mb": values[3], "driver": values[4]})
        report["nvidia_smi"] = {"path": nvidia, "exit": code, "gpus": rows, "raw_available": bool(text)}
    try:
        import torch
        available = bool(torch.cuda.is_available())
        devices = int(torch.cuda.device_count())
        report["torch"] = {
            "version": torch.__version__,
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "available": available,
            "device_count": devices,
            "names": [str(torch.cuda.get_device_name(i)) for i in range(devices)] if available else [],
        }
    except Exception as exc:
        report["torch"] = {"error": type(exc).__name__}
    return report


def imports_report() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            result[name] = {"ok": True, "version": getattr(module, "__version__", None)}
        except Exception as exc:
            result[name] = {"ok": False, "error": type(exc).__name__}
    return result


def competition_schema(data_root: Path, min_test_rows: int = 50, allow_smoke: bool = False) -> dict[str, Any]:
    target = data_root / "input" / "competition" / "name-that-llm"
    train = input_manifest.find_name(target, "train.csv", data_root)
    test = input_manifest.find_name(target, "test.csv", data_root)
    train_ok, train_detail = input_manifest.csv_status(train, {"id", "prompt_id", "text", "model"}, data_root)
    test_ok, test_detail = input_manifest.csv_status(test, {"id", "text"}, data_root, min_rows=1)
    test_rows = int(test_detail.get("rows", 0))
    test_size_ok = allow_smoke or test_rows >= min_test_rows
    test_detail["minimum_rows_for_strict"] = min_test_rows
    test_detail["allow_smoke"] = allow_smoke
    test_detail["full_size_gate"] = test_size_ok
    labels: set[str] = set()
    if train is not None and train_ok:
        try:
            with train.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                labels = {str(row.get("model", "")) for row in reader if row.get("model")}
        except Exception:
            labels = set()
    return {
        "ok": train_ok and test_ok and test_size_ok and labels == EXPECTED_LABELS,
        "train": train_detail,
        "test": test_detail,
        "labels_exact": labels == EXPECTED_LABELS,
        "label_count": len(labels),
    }


def pool_schema(data_root: Path) -> dict[str, Any]:
    target = data_root / "input" / "nishantdahal" / "ntl-two-stage-lora"
    pool = input_manifest.find_name(target, "pool_v1.parquet", data_root)
    if pool is None:
        return {"ok": False, "reason": "pool_v1.parquet missing"}
    try:
        import pandas as pd
        frame = pd.read_parquet(pool, columns=["text", "model"]).head(1)
        return {"ok": {"text", "model"}.issubset(frame.columns) and len(frame) > 0, "columns": list(frame.columns), "rows_sampled": len(frame), "path": str(pool)}
    except Exception as exc:
        return {"ok": False, "reason": type(exc).__name__, "path": str(pool)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", Path.home() / "name-that-llm-data")))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=80.0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--min-test-rows", type=int, default=50)
    parser.add_argument("--allow-smoke", action="store_true", help="Allow a small visible test fixture; never use for a leaderboard submission")
    args = parser.parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    if not data_root.is_dir():
        print(f"DATA_ROOT is not a directory: {data_root}", file=sys.stderr)
        return 2
    config_path = input_manifest.DEFAULT_CONFIG
    config = input_manifest.load_json(config_path)
    verification = input_manifest.verify_config(config, data_root)
    imports = imports_report()
    gpu = gpu_report()
    disk = shutil.disk_usage(data_root)
    disk_report = {"free_gb": round(disk.free / 2**30, 2), "total_gb": round(disk.total / 2**30, 2), "min_free_gb": args.min_free_gb, "ok": disk.free >= args.min_free_gb * 2**30}
    pool = pool_schema(data_root)
    if args.min_test_rows < 1:
        print("--min-test-rows must be positive", file=sys.stderr)
        return 2
    competition = competition_schema(data_root, min_test_rows=args.min_test_rows, allow_smoke=args.allow_smoke)
    credentials = input_manifest.credential_source()
    downloader_available = bool(shutil.which("kaggle"))
    try:
        importlib.import_module("kagglehub")
        downloader_available = True
    except Exception:
        pass
    torch_ok = bool(gpu.get("torch", {}).get("available")) and int(gpu.get("torch", {}).get("device_count", 0)) > 0
    imports_ok = all(item.get("ok") for item in imports.values())
    competition_ok = bool(competition["ok"])
    strict_ok = bool(verification["ok"] and competition_ok and imports_ok and torch_ok and disk_report["ok"] and pool["ok"])
    if not verification["ok"] and not downloader_available and credentials == "none":
        strict_ok = False
    report = {
        "schema_version": 1,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_root_recorded": True,
        "python": {"version": platform.python_version(), "executable_recorded": True},
        "platform": platform.platform(),
        "credentials": credentials,
        "downloader_available": downloader_available,
        "imports": imports,
        "gpu": gpu,
        "disk": disk_report,
        "competition_schema": competition,
        "pool_schema": pool,
        "input_verification": verification,
        "strict_ok": strict_ok,
    }
    output = args.json_out.expanduser().resolve() if args.json_out else data_root / "logs" / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"strict_ok": strict_ok, "report_recorded": True, "missing_input_count": len(verification["missing_handles"]), "cuda": torch_ok, "disk_ok": disk_report["ok"], "imports_ok": imports_ok, "competition_ok": competition_ok}, ensure_ascii=False))
    if args.strict and not strict_ok:
        print(f"STRICT PREFLIGHT FAILED; inspect {output}", file=sys.stderr)
        if credentials == "none" and not verification["ok"]:
            print("Missing inputs and no Kaggle credential. Set KAGGLE_API_TOKEN or install ~/.kaggle/kaggle.json.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
