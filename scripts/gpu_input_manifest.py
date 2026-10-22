#!/usr/bin/env python3
"""Download and fail-closed verification for the compact GPU route.

This file is deliberately stdlib-only so it can run before the ML environment is
installed. It never prints or stores Kaggle credentials. It downloads only the
inputs in config/kaggle_inputs.json, verifies the file tree expected by
kaggle_best_two_stage_lora.py, and writes a private manifest under DATA_ROOT.

Examples:
    python scripts/gpu_input_manifest.py download --data-root "$DATA_ROOT"
    python scripts/gpu_input_manifest.py verify --data-root "$DATA_ROOT"
    python scripts/gpu_input_manifest.py write --data-root "$DATA_ROOT" --hash all
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "kaggle_inputs.json"
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "name-that-llm-data"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def under(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_target(data_root: Path, relative: str) -> Path:
    raw_text = str(relative)
    raw = PurePosixPath(raw_text)
    windows_parts = PureWindowsPath(raw_text).parts
    if (
        Path(raw_text).is_absolute()
        or PureWindowsPath(raw_text).is_absolute()
        or raw.is_absolute()
        or ".." in raw.parts
        or ".." in windows_parts
    ):
        raise ValueError(f"Unsafe target path in config: {relative}")
    if not raw.parts or any(part in {"", "."} for part in raw.parts):
        raise ValueError(f"Target must be a non-empty relative path: {relative}")
    target = (data_root / Path(*raw.parts)).resolve()
    root = data_root.resolve()
    if not under(root, target):
        raise ValueError(f"Target escapes DATA_ROOT: {relative}")
    return target


def safe_file(path: Path, root: Path) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        return under(root.resolve(), path.resolve())
    except OSError:
        return False


def iter_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    result: list[Path] = []
    for path in root.rglob("*"):
        if safe_file(path, root):
            result.append(path)
    return sorted(result)


def nonempty(path: Path, root: Path) -> bool:
    return safe_file(path, root) and path.stat().st_size > 0


def suffix_match(path: Path, relative: str) -> bool:
    wanted = tuple(PurePosixPath(relative).parts)
    actual = tuple(path.parts)
    return len(actual) >= len(wanted) and actual[-len(wanted):] == wanted


def find_required(target: Path, relative: str, root: Path) -> Path | None:
    direct = target / Path(*PurePosixPath(relative).parts)
    if nonempty(direct, root):
        return direct
    for path in iter_files(target):
        if suffix_match(path, relative) and nonempty(path, root):
            return path
    return None


def find_name(target: Path, name: str, root: Path) -> Path | None:
    direct = target / name
    if nonempty(direct, root):
        return direct
    for path in iter_files(target):
        if path.name == name and nonempty(path, root):
            return path
    return None


def csv_status(path: Path | None, required: set[str], root: Path, min_rows: int = 1, allow_smoke: bool = False) -> tuple[bool, dict[str, Any]]:
    if path is None or not nonempty(path, root):
        return False, {"reason": "missing_or_empty", "rows": 0}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = [x.strip() for x in next(reader, [])]
            if len(set(header)) != len(header) or not required.issubset(header):
                return False, {"reason": "missing_or_duplicate_headers", "header": header, "rows": 0}
            text_index = header.index("text")
            rows = 0
            nonempty_text = 0
            for row in reader:
                if not any(x.strip() for x in row):
                    continue
                if len(row) != len(header):
                    return False, {"reason": "malformed_row_width", "rows": rows, "header": header}
                rows += 1
                if row[text_index].strip():
                    nonempty_text += 1
            size_ok = allow_smoke or rows >= min_rows
            ok = rows > 0 and nonempty_text > 0 and size_ok
            reason = "ok" if ok else ("too_few_rows" if rows < min_rows and not allow_smoke else "no_nonempty_text_rows")
            return ok, {"reason": reason, "header": header, "rows": rows, "nonempty_text_rows": nonempty_text, "minimum_rows": min_rows, "allow_smoke": allow_smoke}
    except (OSError, UnicodeError, csv.Error) as exc:
        return False, {"reason": "unreadable_" + type(exc).__name__, "rows": 0}


def json_selector_ok(path: Path, selector: dict[str, Any]) -> tuple[bool, str]:
    try:
        value = load_json(path)
        nested = value.get("text_config") if isinstance(value.get("text_config"), dict) else {}
        for key, expected in selector.items():
            actual = value.get(key, nested.get(key))
            if actual != expected:
                return False, f"{key}={actual!r}, expected {expected!r}"
        return True, "selector matched"
    except Exception as exc:
        return False, type(exc).__name__


def required_files_check(item: dict[str, Any], target: Path, root: Path) -> dict[str, Any]:
    required = item.get("required", [])
    found: dict[str, str] = {}
    missing: list[str] = []
    for relative in required:
        path = find_required(target, relative, root)
        if path is None:
            missing.append(relative)
        else:
            found[relative] = str(path)
    return {"ok": not missing, "found": found, "missing": missing}


def required_globs_check(item: dict[str, Any], target: Path, root: Path) -> dict[str, Any]:
    found: dict[str, list[str]] = {}
    missing: list[str] = []
    files = iter_files(target)
    for pattern in item.get("required_globs", []):
        matches = [str(path) for path in files if fnmatch.fnmatch(path.name, pattern) and nonempty(path, root)]
        if not matches:
            missing.append(pattern)
        else:
            found[pattern] = matches[:20]
    return {"ok": not missing, "found": found, "missing": missing}


def verify_item(item: dict[str, Any], data_root: Path) -> dict[str, Any]:
    target = safe_target(data_root, str(item["target"]))
    kind = str(item["kind"])
    result: dict[str, Any] = {"kind": kind, "handle": item["handle"], "target": str(target), "ok": False}
    if kind == "competition":
        train = find_name(target, "train.csv", data_root)
        test = find_name(target, "test.csv", data_root)
        train_ok, train_detail = csv_status(train, {"id", "prompt_id", "text", "model"}, data_root)
        test_ok, test_detail = csv_status(test, {"id", "text"}, data_root)
        result.update({"train": train_detail, "test": test_detail, "train_path": str(train) if train else None, "test_path": str(test) if test else None})
        result["ok"] = train_ok and test_ok
        return result
    if kind == "dataset":
        result.update(required_files_check(item, target, data_root))
        result["ok"] = bool(result["ok"])
        return result
    if kind == "model":
        config = find_name(target, "config.json", data_root)
        tokenizer = find_name(target, "tokenizer.json", data_root)
        weights = [p for p in iter_files(target) if fnmatch.fnmatch(p.name, "*.safetensors") and nonempty(p, data_root)]
        selector_ok, selector_detail = (False, "config missing")
        if config is not None:
            selector_ok, selector_detail = json_selector_ok(config, dict(item.get("selector", {})))
        glob_check = required_globs_check(item, target, data_root)
        result.update({
            "config": str(config) if config else None,
            "tokenizer": str(tokenizer) if tokenizer else None,
            "weights": [str(p) for p in weights[:20]],
            "weight_file_count": len(weights),
            "selector": selector_detail,
            "required_globs": glob_check,
        })
        result["ok"] = bool(config and tokenizer and weights and selector_ok and glob_check["ok"])
        return result
    raise ValueError(f"Unsupported input kind: {kind}")


def verify_config(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    items = [verify_item(item, data_root) for item in config.get("downloads", [])]
    return {
        "route": config.get("route"),
        "items": items,
        "ok": all(item["ok"] for item in items),
        "missing_handles": [item["handle"] for item in items if not item["ok"]],
    }


def credential_source() -> str:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "env_token"
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    config_path = config_dir / "kaggle.json"
    if config_path.is_file() and os.access(config_path, os.R_OK):
        try:
            mode = config_path.stat().st_mode & 0o777
        except OSError:
            return "none"
        return "kaggle_json" if not (mode & 0o077) else "kaggle_json_insecure"
    return "none"


def redact(text: str) -> str:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        text = text.replace(token, "<KAGGLE_API_TOKEN_REDACTED>")
    return text


def append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(redact(text))
        if text and not text.endswith("\n"):
            handle.write("\n")


def safe_member_name(name: str) -> PurePosixPath:
    pure = PurePosixPath(name)
    windows_parts = PureWindowsPath(name).parts
    if Path(name).is_absolute() or PureWindowsPath(name).is_absolute() or pure.is_absolute() or ".." in pure.parts or ".." in windows_parts or any(part == "" for part in pure.parts):
        raise ValueError(f"Unsafe archive member: {name}")
    return pure


def archive_paths(target: Path) -> list[Path]:
    return sorted(
        path for path in target.iterdir()
        if path.is_file() and path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}
    )


def safe_destination(target: Path, relative: Path) -> Path:
    root = target.resolve()
    if target.is_symlink():
        raise ValueError(f"Target directory is a symlink: {target}")
    destination = (target / relative).resolve()
    if not under(root, destination):
        raise ValueError(f"Destination escapes target: {relative}")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Destination traverses a symlink: {current}")
    if destination.exists() and destination.is_symlink():
        raise ValueError(f"Destination is a symlink: {destination}")
    return destination


def safe_extract_zip(archive: Path, target: Path) -> int:
    count = 0
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            pure = safe_member_name(info.filename)
            if info.is_dir():
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink rejected: {info.filename}")
            destination = safe_destination(target, Path(*pure.parts))
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            count += 1
    return count


def safe_extract_tar(archive: Path, target: Path) -> int:
    count = 0
    root = target.resolve()
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            pure = safe_member_name(member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"Archive link rejected: {member.name}")
            destination = safe_destination(target, Path(*pure.parts))
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            with source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            count += 1
    return count


def extract_archives(target: Path, keep: bool, log_path: Path) -> None:
    archives = archive_paths(target)
    for archive in archives:
        try:
            count = safe_extract_zip(archive, target) if archive.suffix.lower() == ".zip" else safe_extract_tar(archive, target)
            append_log(log_path, f"extracted {archive.name}: {count} files")
            if not keep:
                archive.unlink()
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"Could not safely extract {archive}: {exc}") from exc


def copy_tree_without_links(source: Path, target: Path) -> int:
    """Copy a downloader cache tree without following any symlink."""
    if source.is_symlink():
        raise RuntimeError(f"Downloader returned a symlink; refusing: {source}")
    count = 0
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Downloader returned a nested symlink; refusing: {path}")
        relative = path.relative_to(source)
        destination = safe_destination(target, relative)
        if path.is_dir():
            if destination.exists() and not destination.is_dir():
                raise RuntimeError(f"Destination is not a directory: {destination}")
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            count += 1
        else:
            raise RuntimeError(f"Downloader returned an unsupported filesystem entry: {path}")
    return count


def materialize_download(source: Path, target: Path, log_path: Path) -> None:
    source = source.expanduser()
    target = target.expanduser()
    if target.exists() and target.is_symlink():
        raise RuntimeError(f"Download target is a symlink; refusing: {target}")
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        raise RuntimeError(f"Downloader returned a symlink; refusing: {source}")
    source = source.resolve()
    if source == target:
        append_log(log_path, f"kagglehub returned the target directory: {target}")
        return
    if source.is_file():
        destination = safe_destination(target, Path(source.name))
        shutil.copy2(source, destination)
        append_log(log_path, f"copied download {source} -> {destination}")
        return
    if not source.is_dir():
        raise RuntimeError(f"Downloader returned a missing path: {source}")
    count = copy_tree_without_links(source, target)
    append_log(log_path, f"copied directory download {source} -> {target}; files={count}")


def try_kagglehub(item: dict[str, Any], target: Path, log_path: Path) -> bool:
    try:
        import kagglehub  # type: ignore
    except Exception:
        return False
    kind = item["kind"]
    handle = str(item["handle"])
    function_name = {"dataset": "dataset_download", "model": "model_download", "competition": "competition_download"}.get(kind)
    if function_name is None or not hasattr(kagglehub, function_name):
        return False
    function = getattr(kagglehub, function_name)
    try:
        result = function(handle, output_dir=str(target))
    except TypeError:
        result = function(handle)
    except Exception as exc:
        append_log(log_path, f"kagglehub {function_name} failed: {type(exc).__name__}")
        return False
    # Different kagglehub releases either honor output_dir or return a cache
    # path. Materialize returned files when the target is still empty; caller
    # verification decides whether a successful API response is sufficient.
    if result is not None and not iter_files(target):
        try:
            materialize_download(Path(result), target, log_path)
        except Exception as exc:
            append_log(log_path, f"kagglehub materialization failed for {type(exc).__name__}")
            return False
    append_log(log_path, f"kagglehub {function_name} succeeded; result_type={type(result).__name__}")
    return bool(iter_files(target))


def run_cli(command: list[str], log_path: Path) -> bool:
    append_log(log_path, "$ <redacted-command>")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        append_log(log_path, f"launcher error: {exc}")
        return False
    append_log(log_path, result.stdout)
    append_log(log_path, result.stderr)
    append_log(log_path, f"exit={result.returncode}")
    return result.returncode == 0


def download_item(item: dict[str, Any], data_root: Path, log_path: Path, keep_archives: bool) -> bool:
    target = safe_target(data_root, str(item["target"]))
    target.mkdir(parents=True, exist_ok=True)
    if try_kagglehub(item, target, log_path):
        extract_archives(target, keep_archives, log_path)
        if verify_item(item, data_root)["ok"]:
            return True
        append_log(log_path, f"kagglehub output did not verify for {item['handle']}; trying CLI")
    kind = item["kind"]
    handle = str(item["handle"])
    if kind == "competition":
        commands = [["kaggle", "competitions", "download", "-c", handle, "-p", str(target)]]
    elif kind == "dataset":
        commands = [["kaggle", "datasets", "download", "-d", handle, "-p", str(target)]]
    elif kind == "model":
        commands = [
            # Official CLI: the handle is a numbered variation version.
            ["kaggle", "models", "variations", "versions", "download", "-p", str(target), "--untar", handle],
        ]
    else:
        raise ValueError(f"Unsupported download kind: {kind}")
    for command in commands:
        if not run_cli(command, log_path):
            continue
        extract_archives(target, keep_archives, log_path)
        if verify_item(item, data_root)["ok"]:
            return True
        append_log(log_path, "CLI output did not verify; trying next command")
    return False


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_records(data_root: Path, hash_mode: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_files(data_root / "input"):
        relative = path.relative_to(data_root).as_posix()
        record: dict[str, Any] = {"path": relative, "bytes": path.stat().st_size}
        should_hash = hash_mode == "all" or (hash_mode == "metadata" and path.suffix.lower() in {".json", ".csv"})
        if should_hash:
            record["sha256"] = hash_file(path)
        records.append(record)
    return records


def write_manifest(config: dict[str, Any], data_root: Path, report: dict[str, Any], hash_mode: str, log_path: Path) -> Path:
    manifest_path = data_root / "manifest.json"
    payload = {
        "schema_version": 1,
        "written_utc": utc_now(),
        "git_commit": git_commit(),
        "route": config.get("route"),
        "data_root_recorded": True,
        "input_dir_recorded": True,
        "credential_source": credential_source(),
        "downloader": "kagglehub_then_kaggle_cli_model_variation_fallback",
        "hash_mode": hash_mode,
        "log_path_recorded": True,
        "verification": report,
        "files": file_records(data_root, hash_mode),
        "downloads": config.get("downloads", []),
        "optional_routes_not_downloaded_by_default": config.get("optional_routes_not_downloaded_by_default", []),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="manifest.", suffix=".tmp", dir=str(manifest_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, manifest_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return manifest_path


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "download", "write"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--hash", dest="hash_mode", choices=("none", "metadata", "all"), default="metadata")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    config = load_json(args.config.expanduser().resolve())
    data_root.mkdir(parents=True, exist_ok=True)
    log_path = (args.log or (data_root / "logs" / f"input-{int(time.time())}.log")).expanduser().resolve()
    report = verify_config(config, data_root)
    if args.command == "download" and report["ok"]:
        print("All configured inputs already verify; no download needed.", flush=True)
    if args.command == "download" and not report["ok"]:
        if credential_source() == "none":
            print("Required inputs are missing and no Kaggle credential was found.", file=sys.stderr)
            print("Provide KAGGLE_API_TOKEN or a readable ~/.kaggle/kaggle.json (mode 600); never commit it.", file=sys.stderr)
            print("If the competition is not accepted, open its Kaggle page and accept the rules first.", file=sys.stderr)
            return 2
        for item in config.get("downloads", []):
            current = verify_item(item, data_root)
            if current["ok"]:
                continue
            print(f"Downloading {item['kind']} input ...", flush=True)
            if not download_item(item, data_root, log_path, args.keep_archives):
                print("Download failed; inspect the private log", file=sys.stderr)
                return 2
        report = verify_config(config, data_root)
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else data_root / "manifest.json"
    written = write_manifest(config, data_root, report, args.hash_mode, log_path)
    if manifest_path != written:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(written, manifest_path)
    print(json.dumps({"ok": report["ok"], "manifest_recorded": True, "log_recorded": True, "missing_count": len(report["missing_handles"])}, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
