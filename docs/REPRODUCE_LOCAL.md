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

Run the audit:

```powershell
python scripts/audit_dataset.py
```

The audit script currently expects the original project layout (`data/train.csv` and `research/evidence`). For the public release, use it as a source reference or adapt the `DATA` and `OUT` paths to your private input/output directories rather than copying data into Git.

For a simple local full fit, adapt the path arguments of `final_inference.py`:

```powershell
python scripts/final_inference.py --data-dir D:/private/name-that-llm-data --output D:/private/name-that-llm-artifacts/submission.parquet
```

The local visible test is only a three-row smoke test. Do not interpret its predictions as a competition score.
