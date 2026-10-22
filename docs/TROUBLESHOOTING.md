# Troubleshooting the remote GPU route

Use the first concrete fix below; do not silently change the method or ask for
new architecture work while a setup problem remains.

| Symptom | Check | Action |
|---|---|---|
| Kaggle 401/403 | `kaggle --version`; `kaggle competitions files -c name-that-llm` | Install a fresh token, `chmod 600 ~/.kaggle/kaggle.json`, and accept competition rules. Never paste a token into chat or GitHub. |
| GitHub clone works but Kaggle download fails | credential source in `server_preflight.py` | GitHub SSH is unrelated to Kaggle API. Configure `KAGGLE_API_TOKEN` or `KAGGLE_CONFIG_DIR/kaggle.json`. |
| `kaggle` command not found | `python -m pip show kaggle`; `command -v kaggle` | Activate `$DATA_ROOT/.venv`, install `requirements-server.txt`, or use the setup script. |
| Model CLI syntax error | `kaggle --version` and input-download log | The wrapper uses the numbered variation-version command. Consult the [official model-variation version reference](https://github.com/Kaggle/kaggle-cli/blob/f0afa32699d28c97f82691728ada3ed8c16c5abf/skills/references/model_variations_versions.md); do not invent a different model. |
| Only 3 test rows | `wc -l .../test.csv` and manifest | This is a smoke input, not hidden leaderboard data. Check competition access or run the full code-only notebook in Kaggle. |
| `torch.cuda.is_available()` false | `nvidia-smi`; `python -c 'import torch; print(torch.version.cuda)'` | Fix the NVIDIA driver/CUDA-matched PyTorch installation. Do not replace a working CUDA torch with a generic wheel. |
| `No module named pyarrow/safetensors/accelerate` | `python -m pip check` and imports | Activate the venv and rerun setup; record resolved versions. |
| Qwen config mismatch | `server_preflight.py --strict` report | Attach the exact Qwen3 8B/14B model instances. Do not edit the config to bypass the selector. |
| Adapter missing | `docs/INPUT_LAYOUT.md`; manifest missing list | Re-download `nishantdahal/ntl-two-stage-lora`; confirm nested `adapter/adapter_config.json`, `adapter_model.safetensors` and `heads.safetensors`. |
| Label order mismatch | script error mentioning checkpoint labels | Stop. Compare the competition snapshot and adapter metadata; never reorder labels by hand. |
| Pool cannot be read | `python -c 'import pandas as p; print(p.read_parquet(...).columns)'` | Install PyArrow in the active venv and verify `text,model` columns. |
| OOM | run log and `nvidia-smi` | Try `--only-8b` as a separate diagnostic, use a larger GPU/2×T4, or free memory. The script rejects CPU/disk offload; do not treat offload as the same route. |
| Output has wrong row count/order | inspect `run.json` and test path | Ensure `--input-dir` points at the full competition input, not the local 3-row smoke file. |
| SSH disconnect | `tmux ls`; run log path | Reattach with `tmux attach -t ntllm`. A model process interrupted mid-run cannot resume; rerun after fixing. |
| Disk/RAM low | `df -h`, `free -h` | Remove only external caches/runs you have reviewed; keep the input manifest and current run until hashes are recorded. |
| Kaggle/GitHub network timeout | `curl -I`/CLI version and logs | Retry after checking network policy. Do not put credentials in URLs or bypass competition access controls. |
| Direct server submission rejected | Kaggle error text | Code-only competitions may require Kaggle Code. Run the same repository script in a Kaggle notebook, Save Version, and submit there. |
| Historical v4 dependency unavailable | prvsiyan source says branch B/Lightning unavailable | Stop calling the run v4/0.604. Use the compact route or label a new branch-A/Gaussian/ATLAS experiment unscored. |
