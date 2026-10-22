# Release checklist

- [ ] `README.md` explains competition, input attachment, and evidence levels.
- [ ] `AI_AGENT_PROMPT.md` is self-contained and points to the first run.
- [ ] CPU and GPU runbooks are present.
- [ ] No raw train/test data is tracked.
- [ ] No model weights/adapters/public pools are tracked.
- [ ] No private PDFs or bundled dependency trees are tracked.
- [ ] No credentials or tokens are tracked.
- [ ] `git ls-files` contains only intended source/docs/compact evidence.
- [ ] Public source URLs and attribution are listed.
- [ ] No tracked script depends on a machine-specific local workspace, vendored private dependencies, or the private workspace layout.
- [ ] Local scripts accept explicit private data/output paths.
- [ ] The two-stage LoRA script has been syntax-checked.
- [ ] The CPU scripts have been syntax-checked.
- [ ] `python scripts/release_audit.py` passes locally and in the release CI workflow.
- [ ] The GitHub target repository name is confirmed before push.
- [ ] After push, clone the remote into a fresh temporary directory and inspect it.
