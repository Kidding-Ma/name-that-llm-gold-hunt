# Local reproduction

The public release does not ship competition data. Place private inputs outside the repository, for example:

```text
D:/private/name-that-llm-data/train.csv
D:/private/name-that-llm-data/test.csv
```

Install the CPU requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-cpu.txt
```

Run the audit with explicit private paths:

```powershell
python scripts/audit_dataset.py --data-dir D:/private/name-that-llm-data --output-dir D:/private/name-that-llm-artifacts/audit
```

The public scripts accept explicit private input/output paths. For example:

```bash
python scripts/audit_dataset.py --data-dir /private/name-that-llm --output-dir /private/artifacts/audit
python scripts/style_aware_runner.py --data-dir /private/name-that-llm --output-dir /private/artifacts/style
python scripts/final_inference.py --data-dir /private/name-that-llm --output /private/artifacts/submission.parquet
```

Keep the data and generated outputs outside the Git repository. The visible local `test.csv` has only three examples and is a schema smoke test, not the hidden leaderboard test.

For a simple local full fit, adapt the path arguments of `final_inference.py`:

```powershell
python scripts/final_inference.py --data-dir D:/private/name-that-llm-data --output D:/private/name-that-llm-artifacts/submission.parquet
```

The local visible test is only a three-row smoke test. Do not interpret its predictions as a competition score.
