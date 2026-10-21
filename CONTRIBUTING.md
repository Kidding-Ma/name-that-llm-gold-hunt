# Contributing

This is a competition research handoff. Keep changes reproducible and safe.

## Required experiment record

For every new method, record:

- method name and one-line hypothesis;
- exact code version/commit;
- input data/model/checkpoint identifiers;
- prompt-grouped validation protocol;
- local CV and/or Kaggle public/private score;
- GPU model, memory, runtime;
- prediction/output hash where applicable;
- license and attribution for external assets;
- whether the result is independently reproduced or upstream-reported.

## Pull request checklist

- [ ] no data, hidden labels, weights, adapters, tokens, or local outputs added;
- [ ] no absolute machine-specific paths;
- [ ] grouped validation is used;
- [ ] output contract is checked;
- [ ] README/runbook updated;
- [ ] external source and license recorded;
- [ ] score evidence level stated.
