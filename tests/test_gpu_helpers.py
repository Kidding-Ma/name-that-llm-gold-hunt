"""Test portable GPU planning/CLI helpers without PyTorch or model downloads.

Only named pure functions are compiled from the script AST. The actual GPU
implementation still requires a Kaggle/CUDA integration run and leaderboard
submission; these unit tests do not establish model accuracy.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_best_two_stage_lora.py"


def helpers() -> dict:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    wanted = {"gpu_memory_plan", "parse_args"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise AssertionError("Expected helper functions were not found")
    namespace = {"argparse": argparse, "Path": Path, "BALANCE_MIN_ROWS": 50}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace


class GpuPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = helpers()["gpu_memory_plan"]

    def test_large_single_gpu_is_allowed(self) -> None:
        self.assertEqual(self.plan(22.0, [45.0]), {0: 42.5})

    def test_two_t4_cards_can_be_combined(self) -> None:
        self.assertEqual(self.plan(22.0, [15.0, 15.0]), {0: 12.5, 1: 12.5})

    def test_larger_nonzero_device_is_selected(self) -> None:
        self.assertEqual(self.plan(12.0, [5.0, 23.0]), {1: 20.5})

    def test_insufficient_single_gpu_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exceeds usable CUDA memory"):
            self.plan(22.0, [15.0])

    def test_no_gpus_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No CUDA memory"):
            self.plan(1.0, [])

    def test_activation_reserve_is_applied(self) -> None:
        with self.assertRaises(RuntimeError):
            self.plan(10.0, [12.0])


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parse = helpers()["parse_args"]

    def test_default_recipe_is_unchanged(self) -> None:
        args = self.parse([])
        self.assertFalse(args.only_8b)
        self.assertFalse(args.only_14b)
        self.assertFalse(args.no_twin)
        self.assertFalse(args.no_balance)
        self.assertIsNone(args.scores_output)
        self.assertEqual(args.balance_min_rows, 50)

    def test_only_14b_is_available(self) -> None:
        args = self.parse(["--only-14b"])
        self.assertTrue(args.only_14b)
        self.assertFalse(args.only_8b)

    def test_member_selectors_are_mutually_exclusive(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                self.parse(["--only-8b", "--only-14b"])
        self.assertEqual(caught.exception.code, 2)

    def test_optional_private_scores_path(self) -> None:
        target = Path("private-experiment") / "scores.npz"
        args = self.parse(["--scores-output", str(target)])
        self.assertEqual(args.scores_output, target)

    def test_invalid_output_arguments_are_rejected(self) -> None:
        cases = [
            ["--balance-min-rows", "0"],
            ["--scores-output", "scores.json"],
            ["--output", "same.parquet", "--metadata", "same.parquet"],
            ["--output", "same.npz", "--scores-output", "same.npz"],
            ["--metadata", "same.npz", "--scores-output", "same.npz"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    self.parse(argv)
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
