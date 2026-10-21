"""Preflight check for the LoRA + Adaptive Centering Kaggle notebook.

Paste this into a Kaggle Notebook before running the full model. It does not
train or predict; it reports whether the expected competition data, model
inputs, datasets, and public notebook outputs appear under /kaggle/input.
"""
from pathlib import Path
import json
import os

ROOT = Path("/kaggle/input")
print("/kaggle/input exists:", ROOT.exists())
print("CUDA visible:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "gpu_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(" GPU", i, torch.cuda.get_device_name(i))
except Exception as exc:
    print("torch check failed:", repr(exc))


def files(pattern):
    return sorted(ROOT.rglob(pattern)) if ROOT.exists() else []


def dirs_containing(fragment):
    return sorted({p.parent for p in files("config.json") if fragment.lower() in str(p.parent).lower()})

print("\nCompetition data:")
for p in files("train.csv"):
    if (p.parent / "test.csv").exists():
        print(" OK", p.parent)
if not any((p.parent / "test.csv").exists() for p in files("train.csv")):
    print(" MISSING train.csv + test.csv pair")

print("\nExpected external data files:")
for pattern in ["ntl_extra_passages_claude.csv", "gen_passages.csv", "ntl_generations.csv", "ntl_proxy_generations.csv"]:
    hs = files(pattern)
    print(" OK" if hs else " MISSING", pattern, hs[:3])

print("\nExpected model directories/configs:")
for label, fragments in {
    "Qwen3 8B": ["qwen-3", "8b"],
    "Qwen3 4B": ["qwen-3", "4b"],
    "Qwen2.5 7B": ["qwen2.5", "7b"],
    "Qwen3.5 9B": ["qwen35", "qwen-3.5"],
    "UltraBERT": ["ultrabert"],
    "DeBERTa": ["deberta"],
    "ELECTRA": ["electra"],
}.items():
    matches = []
    for cfg in files("config.json"):
        s = str(cfg.parent).lower()
        if all(f.lower() in s for f in fragments):
            matches.append(cfg.parent)
    print(" OK" if matches else " CHECK", label, matches[:3])

print("\nExpected public kernel-output markers:")
for pattern in [
    "**/lora24/final/COMPLETE.json",
    "**/lora_out/job_final/COMPLETE.json",
    "**/tokattn_final.pt",
    "**/tokattn_debl23_final.pt",
    "**/heads.safetensors",
    "**/anchor_features.npz",
    "**/norms.safetensors",
    "**/valid_features.npz",
]:
    hs = files(pattern)
    print(" OK" if hs else " MISSING", pattern, hs[:2])

print("\nIf several lines say MISSING, do not run the full notebook yet.")
print("Add the missing sources listed in KAGGLE_LORA_SUBMIT_GUIDE_zh.md, then rerun this check.")
