"""Static smoke tests for server handoff tools; no network, credentials or GPU."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import importlib.util
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ServerToolTests(unittest.TestCase):
    def test_input_manifest_is_valid_json_and_compact_only(self) -> None:
        config = json.loads((ROOT / "config" / "kaggle_inputs.json").read_text(encoding="utf-8"))
        self.assertEqual(config["route"], "nishant_compact_control")
        self.assertEqual(len(config["downloads"]), 4)
        self.assertTrue(all(item["target"].startswith("input/") for item in config["downloads"]))
        self.assertNotIn("lightning", " ".join(config["optional_routes_not_downloaded_by_default"]).lower())

    def test_python_tools_compile(self) -> None:
        paths = [
            ROOT / "scripts" / "gpu_input_manifest.py",
            ROOT / "scripts" / "server_preflight.py",
        ]
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def load_manifest_module(self):
        spec = importlib.util.spec_from_file_location("gpu_input_manifest_test", ROOT / "scripts" / "gpu_input_manifest.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_csv_smoke_gate_defaults_to_full_size(self) -> None:
        module = self.load_manifest_module()
        root = ROOT / ".server-tool-virtual-root"
        path = root / "test.csv"
        fake_files = {path: "id,text\n1,smoke\n"}
        with patch.object(Path, "is_file", lambda candidate: candidate in fake_files), \
             patch.object(Path, "is_symlink", lambda candidate: False), \
             patch.object(Path, "resolve", lambda candidate: candidate), \
             patch.object(Path, "stat", lambda candidate: SimpleNamespace(st_size=len(fake_files[candidate].encode("utf-8")))), \
             patch.object(Path, "open", lambda candidate, *args, **kwargs: io.StringIO(fake_files[candidate])):
            ok, detail = module.csv_status(path, {"id", "text"}, root, min_rows=50)
            self.assertFalse(ok)
            self.assertEqual(detail["reason"], "too_few_rows")
            ok, detail = module.csv_status(path, {"id", "text"}, root, min_rows=50, allow_smoke=True)
            self.assertTrue(ok)

    def test_empty_root_fails_closed_without_credentials(self) -> None:
        module = self.load_manifest_module()
        root = ROOT / ".server-tool-virtual-root"
        config = json.loads((ROOT / "config" / "kaggle_inputs.json").read_text(encoding="utf-8"))
        with patch.object(Path, "is_dir", lambda candidate: False), \
             patch.object(Path, "is_file", lambda candidate: False), \
             patch.object(Path, "is_symlink", lambda candidate: False):
            report = module.verify_config(config, root)
            self.assertFalse(report["ok"])
            self.assertEqual(len(report["missing_handles"]), 4)


if __name__ == "__main__":
    unittest.main()
