# Controlled cleaning and blocking comparison

This is the historical cleaning-stage report. The subsequent
[model comparison and frozen evaluation](model_comparison.md) is now complete;
evaluation is no longer an untouched holdout. The stage-specific results below
remain preserved.

The original dataset and saved `blocking_v1` outputs are preserved. This experiment
compares derived text views at fixed K=10, with the same character 3–5 gram TF-IDF,
cosine retrieval, source/country partitions, and deterministic tie handling.
Each variant's vocabulary and IDF are refitted on training records only. Audit
tables and rule definitions were saved before opening validation records.

## Audited rules

The training audit found 1,062 address placeholder components and six explicit
`<CITY_NAME>` name markers. Placeholder cleanup changes 1,068 training field views.
Examples of real names containing `Null` or `N/A` are retained. Unicode
non-joiners are also retained, rather than treated as noise.

- Baseline: existing NFKC/case/punctuation/whitespace normalization.
- Placeholders: remove whole comma-delimited address components `NULL`, `null`,
  `N/A`, `<NULL>`; remove explicit embedded `<NULL>` in addresses and
  `<CITY_NAME>` in names. Preserve the rest of each field.
- Placeholders plus abbreviations: additionally expand name tokens pvt/ltd/inc/corp
  and address tokens rd/ave/blvd/apt/ste using the frozen field-specific map.

`Null Industries`, `N/A Science Inc`, `Null Road`, accents, non-Latin text, and
joiners/non-joiners are protected. Country fields, ambiguous St/No tokens, typo
correction, transliteration, and generic legal-suffix deletion are outside this test.
The abbreviation map is a measured heuristic, not a universal semantic guarantee.

## Measured validation results

All variants retain **the same 5,182 of 5,189 true pairs**; none recovers or newly
loses a validation true pair.

| Retrieval text | Recall | Candidates | Added candidates | Removed candidates |
|---|---:|---:|---:|---:|
| Baseline | 99.86510% | 55,430 | 0 | 0 |
| Placeholders | 99.86510% | 55,426 | 319 | 323 |
| Placeholders + abbreviations | 99.86510% | 55,425 | 4,871 | 4,876 |

The net count change is tiny, but candidate identities do change. In particular,
abbreviation expansion swaps thousands of nonmatches in the validation pool.
Such changes may matter to a future classifier even with identical blocking recall.
They do not establish a precision/F0.5 improvement without model decisions.

## Training diagnostics (in-sample)

| Retrieval text | Recall | Recovered true pairs | Newly lost true pairs |
|---|---:|---:|---:|
| Baseline | 99.33127% | 0 | 0 |
| Placeholders | 99.33953% | 4 | 2 |
| Placeholders + abbreviations | 99.31889% | 7 | 10 |

The training corpus is used to fit the representation, so these are diagnostics,
not held-out estimates. Offsetting gains and losses show why reporting only net
recall would conceal regressions. Neither cleaning variant shows a broad recall
improvement across this first experiment.

## Decision

Keep the original baseline adopted and preserve both cleaning branches for later
controlled model comparisons. The artifact's minimum-count suggestion chooses the
abbreviation variant by a one-candidate tie-break; that is a mechanical shortlist,
not meaningful evidence of superiority. Do not promote a cleaning variant based
on this tiny candidate-count difference.

The bounded comparison is complete; no extra rule tuning was performed after
seeing validation outcomes. Evaluation remains unopened, and no logistic
regression or XGBoost classifier has been trained. Results apply to the reduced
candidate pool; final matching and full-pool performance remain unmeasured.

## Outputs and reproduction

- [Executed notebook](../notebooks/03_cleaning_ablation.ipynb)
- [Offline interactive report](../reports/cleaning/cleaning_ablation.html)
- [Findings](../reports/cleaning/findings.md)
- `artifacts/cleaning_v1/`: audit tables, frozen rules, candidate sets and evidence
  for all three variants in training/validation, paired changes, missed pairs,
  fitted vectorizers, metrics, and input/artifact hashes.

From `lab_project/`, rebuild visualizations from saved results:

```powershell
uv run python scripts/build_cleaning_report.py
uv run jupyter lab notebooks/03_cleaning_ablation.ipynb
uv run python -m unittest discover -s scripts -p 'test_*.py' -v
```

Repeat the experiment into a new directory, preserving the previous one:

```powershell
uv run python scripts/run_cleaning.py --output-dir artifacts/cleaning_repeat
uv run python scripts/build_cleaning_report.py --run-dir artifacts/cleaning_repeat
```

The notebook authoring source is `scripts/build_cleaning_report.py`; rebuilding
overwrites its generated notebook/report. The experiment refuses to overwrite
existing artifacts and marks failures with `INCOMPLETE.txt`. It needs the verified
saved `blocking_v1` baseline. No new dependencies were needed; the existing uv
environment and lockfile are reused.
