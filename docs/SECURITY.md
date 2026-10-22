# Security and privacy

## Never commit

- Kaggle API keys, cookies, session files, or credentials;
- GitHub tokens, SSH private keys, cloud credentials, or `.env` files;
- competition train/test CSV files or hidden labels;
- raw prompts/responses copied from private artifacts;
- model weights, LoRA adapters, tokenizer caches, or public pools;
- local logs containing absolute home paths or access URLs.

## Before pushing

```bash
git status --short
git ls-files
```

Inspect the file list and search tracked text for credentials and local paths. Keep large artifacts outside GitHub and attach them as Kaggle inputs.

## Competition integrity

This repository is intended for legal competition research. Do not manually label hidden test rows, use hidden test labels, or share private competition material in issues and pull requests. Follow the competition rules and all third-party dataset/model licenses.

## Reporting a problem

If a credential or private dataset is accidentally committed, revoke it first, remove it from Git history, and notify the repository owner. Do not merely delete the working-tree file.
