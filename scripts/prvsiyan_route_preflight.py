"""Read-only, fail-closed screening of the public GPU route's current inputs.

The dataset/model list below is from the CURRENT v10 pull metadata, not a
confirmed inventory of historical v4. The author/source claim for v4 is 0.604
(scriptVersionId=360252286, submission 57428928); it is not our reproduction.
Original v4 source and input versions still need confirmation. Current v10
explicitly disables historical branch B because its Lightning checkpoint is
unavailable, and calls itself an unscored candidate. This screening cannot
establish historical v4 dependency completeness, even when it returns zero.

This script never trains, predicts, downloads, or writes files. Dataset names
are path-name heuristics. CSV structure, nonempty marker files, JSON structure,
model dimensions and visible CUDA are screened, but exact input versions,
checkpoint contents, licenses, memory capacity and score are NOT validated.

Usage: python prvsiyan_route_preflight.py --input-dir /kaggle/input
Or paste this file into a Kaggle notebook cell.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable


ROUTE = {
    "notebook": "https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author",
    "version_url": "https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286",
    "metadata_version": 10,
    "reported_public_ba": 0.604,
    "completed_submission": "57428928",
    "datasets": [
        "josephadamski91/ntl-atlas-qwen3-8b-v1",
        "prvsiyan/ntl-public-in-style-pool-v1",
        "prvsiyan/ntl-two-stage-lora-adapters-v1",
        "nishantdahal/ntl-two-stage-lora",
    ],
    "models": [
        "qwen-lm/qwen-3/Transformers/8b-base/1",
        "qwen-lm/qwen-3/Transformers/14b-base/1",
    ],
}
ADAPTER_FILES = (
    "meta.json", "adapter/adapter_config.json", "adapter/adapter_model.safetensors", "heads.safetensors"
)
NISHANT_SLUG = "ntl-two-stage-lora"


def all_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


def nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def path_matches(paths: Iterable[Path], *fragments: str) -> list[Path]:
    """Path substring heuristic only; never an exact identity/version check."""
    needles = tuple(x.lower() for x in fragments)
    return sorted({p.parent for p in paths if all(n in str(p).lower() for n in needles)})


def named_mount_files(paths: Iterable[Path], slug: str) -> list[Path]:
    """Require the slug as a path component, still only a name heuristic."""
    return [p for p in paths if slug.lower() in (part.lower() for part in p.parts)]


def _csv_header(path: Path) -> set[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return {item.strip() for item in next(csv.reader(handle, strict=True), [])}
    except (OSError, UnicodeError, csv.Error):
        return set()


def csv_status(path: Path | None, required: set[str]) -> tuple[bool, str]:
    if path is None or not nonempty_file(path):
        return False, "CSV absent or empty"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = [item.strip() for item in next(reader, [])]
            if len(set(header)) != len(header) or not required.issubset(header):
                return False, "required unique CSV headers: " + ",".join(sorted(required))
            text_index = header.index("text")
            for row in reader:
                if not any(item.strip() for item in row):
                    continue
                if len(row) != len(header):
                    return False, "malformed CSV data-row width"
                if row[text_index].strip():
                    return True, "required headers and a nonempty text row visible"
            return False, "CSV contains no nonempty text data rows"
    except (OSError, UnicodeError, csv.Error) as exc:
        return False, "CSV unreadable: " + type(exc).__name__


def find_competition_pair(paths: list[Path]) -> tuple[Path | None, Path | None]:
    for train in sorted(p for p in paths if p.name == "train.csv"):
        test = train.parent / "test.csv"
        if csv_status(train, {"id", "prompt_id", "text", "model"})[0] and csv_status(
            test, {"id", "text"}
        )[0]:
            return train, test
    return None, None


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def qwen_configs(paths: list[Path]) -> dict[str, list[Path]]:
    """Select Qwen3 architecture dimensions, not exact base-model versions."""
    found: dict[str, list[Path]] = {"8b": [], "14b": []}
    for cfg_path in paths:
        if cfg_path.name != "config.json":
            continue
        cfg = read_json(cfg_path)
        nested = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
        model_type = cfg.get("model_type", nested.get("model_type"))
        hidden = cfg.get("hidden_size", nested.get("hidden_size"))
        layers = cfg.get("num_hidden_layers", nested.get("num_hidden_layers"))
        if model_type != "qwen3":
            continue
        if hidden == 4096 and layers == 36:
            found["8b"].append(cfg_path.parent)
        if hidden == 5120 and layers == 40:
            found["14b"].append(cfg_path.parent)
    return found


def cuda_status() -> tuple[bool, str]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
        if not available or count < 1:
            return False, f"torch CUDA available={available}; GPU count={count}"
        names = [str(torch.cuda.get_device_name(i)) for i in range(count)]
        return True, f"torch {torch.__version__}; GPU count={count}; " + "; ".join(names)
    except Exception as exc:
        return False, "torch/CUDA check failed: " + type(exc).__name__


def adapter_checks(paths: list[Path], directory: str) -> list[tuple[str, bool, str]]:
    """Require one coherent, nonempty adapter bundle under the named mount."""
    mounted = named_mount_files(paths, NISHANT_SLUG)
    roots = sorted({ancestor for path in mounted for ancestor in path.parents
                    if ancestor.name == directory})
    if roots:
        root = max(roots, key=lambda p: sum(nonempty_file(p / name) for name in ADAPTER_FILES))
    else:
        root = None
    checks = []
    for name in ADAPTER_FILES:
        path = root / name if root is not None else None
        ok = path is not None and nonempty_file(path)
        if ok and name.endswith(".json"):
            try:
                ok = isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
            except (OSError, UnicodeError, ValueError):
                ok = False
        checks.append((f"adapter.{directory}.{name}", bool(ok),
                       "nonempty file/JSON object present" if ok else "missing, empty or invalid required file"))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/kaggle/input"))
    args = parser.parse_args(argv)
    root = args.input_dir.expanduser().resolve()
    checks: list[tuple[str, bool, str]] = []
    try:
        paths = all_paths(root)
        checks.append(("input.inventory", root.is_dir(), "read-only inventory"))
    except OSError as exc:
        paths = []
        checks.append(("input.inventory", False, type(exc).__name__))

    print("Public GPU route screening: current v10 metadata, NOT confirmed original v4 assets")
    print("Notebook:", ROUTE["notebook"])
    print("Historical v4 score-claim URL:", ROUTE["version_url"])
    print("Evidence: author/source text; not an independent reproduction")
    print("HISTORICAL LIMITATION: v10 disables unavailable branch B and is unscored.")
    print("This screening cannot certify historical v4 dependency completeness or parity.")
    print("Input root:", root)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
    ok, detail = cuda_status()
    checks.append(("cuda", ok, detail))

    train, test = find_competition_pair(paths)
    if train is None:
        # Retain separate failure categories even when a pair is malformed.
        candidates = sorted({p.parent for p in paths if p.name in {"train.csv", "test.csv"}})
        if candidates:
            train, test = candidates[0] / "train.csv", candidates[0] / "test.csv"
    for category, path, required in (
        ("competition.train_csv", train, {"id", "prompt_id", "text", "model"}),
        ("competition.test_csv", test, {"id", "text"}),
    ):
        ok, detail = csv_status(path, required)
        checks.append((category, ok, detail))

    print("Dataset checks below are PATH-NAME HEURISTICS, not content/version validation.")
    for identifier in ROUTE["datasets"]:
        slug = identifier.rsplit("/", 1)[-1]
        hits = [p for p in named_mount_files(paths, slug) if nonempty_file(p)]
        checks.append(("dataset." + identifier, bool(hits), "path-name heuristic; nonempty file required"))

    pool = [p for p in named_mount_files(paths, NISHANT_SLUG)
            if p.name == "pool_v1.parquet" and nonempty_file(p)]
    checks.append(("pool.pool_v1.parquet", bool(pool), "nonempty named marker only; Parquet not decoded"))
    for directory in ("lora8b_v1", "lora14b_v1"):
        checks.extend(adapter_checks(paths, directory))
    qwen = qwen_configs(paths)
    for size in ("8b", "14b"):
        checks.append(("model.qwen3_" + size + "_config", bool(qwen[size]),
                       "architecture selector only; weights/base/version not validated"))

    for category, ok, detail in checks:
        print(f"{'OK' if ok else 'MISSING':8} [{category}] {detail}")
    missing = [category for category, ok, _ in checks if not ok]
    print("\nMissing categories:", ", ".join(missing) if missing else "none")
    if missing:
        print("FAIL: resolve missing categories before running this GPU route.")
        return 2
    print("SCREENING PASS only: confirm native v4 source/input versions, licenses, full model")
    print("weights, GPU memory and checkpoint contents manually before claiming reproduction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main([] if "ipykernel" in sys.modules else None))
