# Release policy

This repository is public by design. Keep it small, auditable, and safe for collaboration.

## Allowed

- source code that reads data from an explicit input path;
- compact metrics and configuration summaries;
- public source URLs and attribution;
- runbooks and AI handoff prompts;
- small label/error tables without raw text.

## Forbidden

- competition train/test files;
- sample submissions or predictions containing hidden-test rows;
- model weights, LoRA adapters, tokenizer caches, or public pools;
- PDFs copied from third parties unless licensing and size are appropriate;
- Kaggle/GitHub credentials, environment files, cookies, or tokens;
- absolute local paths in new scripts;
- raw prompts/responses copied into issues or logs.

## Before every push

1. Run `git status --short`.
2. Run `git ls-files` and inspect the list.
3. Search tracked files for `D:\Kaggle`, `C:\Users`, tokens, passwords, and private URLs.
4. Check that no file is unexpectedly large.
5. Confirm that all third-party references are linked in `docs/PUBLIC_SOURCES.md`.
6. Confirm the README still explains how to attach data and checkpoints.

## External artifacts

Store large artifacts in Kaggle datasets, Hugging Face, or a private object store. Record the public identifier, version, license, and expected file names in the docs; never commit the binary artifact itself.
