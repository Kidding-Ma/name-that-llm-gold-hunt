"""Dependency-free synthetic preflight tests; no downloads, GPU or real data.

Path I/O is virtualized: this Windows sandbox denies access to directories made
by tempfile, even under the workspace. Fixtures never write checkpoint markers,
CSV data or files; mocked path operations exercise the same screening functions.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "route_preflight", ROOT / "scripts" / "prvsiyan_route_preflight.py"
)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def fake_torch(available: bool = True, count: int = 1):
    return types.SimpleNamespace(
        __version__="mock-only",
        cuda=types.SimpleNamespace(
            is_available=Mock(return_value=available),
            device_count=Mock(return_value=count),
            get_device_name=Mock(return_value="mock GPU"),
        ),
    )


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / ".virtual-preflight-fixture"
        self.files = {}
        self.dirs = {self.root}
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(Path, "is_file", lambda p: p in self.files))
        self.stack.enter_context(patch.object(Path, "is_dir", lambda p: p in self.dirs))
        self.stack.enter_context(patch.object(Path, "stat", lambda p: types.SimpleNamespace(
            st_size=len(self.files.get(p, "").encode("utf-8")))))
        self.stack.enter_context(patch.object(Path, "rglob", lambda p, pattern: iter(
            path for path in self.files if p in path.parents)))

        def read_text(path, *args, **kwargs):
            if path not in self.files:
                raise FileNotFoundError(path)
            return self.files[path]

        def open_text(path, *args, **kwargs):
            text = read_text(path)
            if kwargs.get("encoding") == "utf-8-sig":
                text = text.removeprefix("\ufeff")
            return io.StringIO(text)

        self.stack.enter_context(patch.object(Path, "read_text", read_text))
        self.stack.enter_context(patch.object(Path, "open", open_text))

    def put(self, relative, text="synthetic-marker"):
        path = self.root / relative
        self.files[path] = text
        self.dirs.update(path.parents)
        return path

    def complete_inputs(self):
        self.put("name-that-llm/train.csv", 'id,prompt_id,text,model\n1,p,"synthetic, text",m\n')
        self.put("name-that-llm/test.csv", 'id,text\n2,"synthetic test"\n')
        for identifier in PREFLIGHT.ROUTE["datasets"]:
            self.put(identifier.rsplit("/", 1)[-1] + "/synthetic-marker.txt")
        self.put("ntl-two-stage-lora/pool_v1.parquet")
        for directory in ("lora8b_v1", "lora14b_v1"):
            for name in PREFLIGHT.ADAPTER_FILES:
                self.put("ntl-two-stage-lora/" + directory + "/" + name,
                         "{}" if name.endswith(".json") else "dummy-weight-marker")
        for size, hidden, layers in (("8b", 4096, 36), ("14b", 5120, 40)):
            self.put("qwen-" + size + "/config.json", json.dumps({
                "model_type": "qwen3", "hidden_size": hidden, "num_hidden_layers": layers
            }))

    def run_main(self, torch_module):
        output = io.StringIO()
        with patch.dict(sys.modules, {"torch": torch_module}), contextlib.redirect_stdout(output):
            result = PREFLIGHT.main(["--input-dir", str(self.root)])
        return result, output.getvalue()

    def test_missing_inputs_list_explicit_categories(self):
        result, output = self.run_main(fake_torch())
        self.assertEqual(result, 2)
        self.assertIn("Missing categories:", output)
        self.assertIn("competition.train_csv", output)
        self.assertIn("competition.test_csv", output)
        self.assertIn("adapter.lora8b_v1.adapter/adapter_model.safetensors", output)
        self.assertIn("model.qwen3_14b_config", output)

    def test_all_visible_inputs_are_only_screening_pass(self):
        self.complete_inputs()
        result, output = self.run_main(fake_torch())
        self.assertEqual(result, 0)
        self.assertIn("SCREENING PASS only", output)
        self.assertIn("current v10 metadata, NOT confirmed original v4 assets", output)
        self.assertIn("PATH-NAME HEURISTICS", output)
        self.assertIn("Missing categories: none", output)

    def test_no_cuda_fails_even_with_all_files(self):
        self.complete_inputs()
        result, output = self.run_main(fake_torch(False, 0))
        self.assertEqual(result, 2)
        self.assertIn("Missing categories: cuda", output)

    def test_cuda_true_with_zero_devices_fails(self):
        self.complete_inputs()
        result, output = self.run_main(fake_torch(True, 0))
        self.assertEqual(result, 2)
        self.assertIn("Missing categories: cuda", output)

    def test_torch_absent_fails(self):
        self.complete_inputs()
        result, output = self.run_main(None)
        self.assertEqual(result, 2)
        self.assertIn("torch/CUDA check failed", output)
        self.assertIn("Missing categories: cuda", output)

    def test_required_head_marker_cannot_be_replaced_by_metadata(self):
        self.complete_inputs()
        self.put("ntl-two-stage-lora/lora14b_v1/heads.safetensors", "")
        result, output = self.run_main(fake_torch())
        self.assertEqual(result, 2)
        self.assertIn("Missing categories: adapter.lora14b_v1.heads.safetensors", output)

    def test_invalid_adapter_config_fails(self):
        self.complete_inputs()
        self.put("ntl-two-stage-lora/lora8b_v1/adapter/adapter_config.json", "not-json")
        result, output = self.run_main(fake_torch())
        self.assertEqual(result, 2)
        self.assertIn("Missing categories: adapter.lora8b_v1.adapter/adapter_config.json", output)

    def test_quoted_csv_headers_and_bom_are_parsed(self):
        path = self.put("quoted.csv", '\ufeff"id","prompt_id","text","model"\n1,p,"a, b",m\n')
        self.assertEqual(PREFLIGHT._csv_header(path), {"id", "prompt_id", "text", "model"})
        self.assertTrue(PREFLIGHT.csv_status(path, {"id", "text"})[0])

    def test_header_only_csv_is_not_nonempty_data(self):
        path = self.put("empty.csv", "id,text\n")
        self.assertFalse(PREFLIGHT.csv_status(path, {"id", "text"})[0])

    def test_wrong_text_header_is_rejected(self):
        path = self.put("wrong.csv", "id,response\n1,synthetic\n")
        self.assertFalse(PREFLIGHT.csv_status(path, {"id", "text"})[0])

    def test_empty_text_and_malformed_width_rejected(self):
        blank = self.put("blank.csv", "id,text\n1,\n")
        wrong = self.put("width.csv", "id,text\n1,synthetic,extra\n")
        self.assertFalse(PREFLIGHT.csv_status(blank, {"id", "text"})[0])
        self.assertFalse(PREFLIGHT.csv_status(wrong, {"id", "text"})[0])

    def test_valid_pair_requires_both_data_rows(self):
        self.complete_inputs()
        self.put("name-that-llm/test.csv", "id,text\n")
        self.assertEqual(PREFLIGHT.find_competition_pair(PREFLIGHT.all_paths(self.root)), (None, None))

    def test_qwen_config_selectors_reject_wrong_architecture(self):
        self.put("bad/config.json", json.dumps({
            "model_type": "qwen2", "hidden_size": 4096, "num_hidden_layers": 36
        }))
        self.put("bad-dim/config.json", json.dumps({
            "model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 40
        }))
        self.assertEqual(PREFLIGHT.qwen_configs(PREFLIGHT.all_paths(self.root)), {"8b": [], "14b": []})

    def test_qwen_config_selectors_accept_nested_dimensions(self):
        path = self.put("nested/config.json", json.dumps({"text_config": {
            "model_type": "qwen3", "hidden_size": 5120, "num_hidden_layers": 40
        }}))
        self.assertEqual(PREFLIGHT.qwen_configs([path]), {"8b": [], "14b": [path.parent]})

    def test_slug_substring_is_not_a_named_mount(self):
        path = self.put("not-ntl-two-stage-lora/wrong.txt")
        self.assertEqual(PREFLIGHT.named_mount_files([path], "ntl-two-stage-lora"), [])


if __name__ == "__main__":
    unittest.main()
