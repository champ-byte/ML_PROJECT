# ML lab project

All lab code and generated data live here. The original competition package,
including its full datasets, documentation, and submission validator, stays in
`../student_resource/`.

The [algorithm selection and visualization design](docs/algorithm_selection.md)
documents proposed blocking/matching comparisons, competition-compatible metrics,
and research references. A reusable [BibTeX bibliography](docs/references.bib) is included.
Candidate generation, cleaning, and six matching pipelines have been compared.
The validation-selected pipeline has completed its single frozen evaluation pass.

## Matching models and interactive walkthrough

Start with the [interactive walkthrough](reports/model_comparison/interactive_walkthrough.html):
choose a validation business/pipeline, step through records → bucketing/TF-IDF →
features → learned weights → decisions, inspect contributions, and adjust the validation threshold. Its final
evaluation readout remains fixed. The page works offline and deliberately includes
diagnostic error examples; global metrics cover all validation businesses.
The bucketing view shows worked fragment weights, top-10 lists, and their union.
The weights view shows actual logistic coefficients or saved XGBoost split paths
and leaf outputs, and how they add up to a score. These views do not retrain models
or rerun evaluation.

The [side-by-side comparison](reports/model_comparison/side_by_side_comparison.html)
shows both models' scores, thresholds, base terms, and feature contributions for
the same candidate. Switch cleaning variants to compare logistic coefficients
(standardized or raw units). Overall contribution magnitudes use the same 50,501
validation candidate pairs across all variants. XGBoost explanations are SHAP
contributions from trees, rather than fixed feature coefficients. Candidates absent
from a variant are explicitly marked as not retrieved and not scored.

See the executed [feature notebook](notebooks/04_pair_features.ipynb),
[model-comparison notebook](notebooks/05_model_comparison.ipynb),
[comparison report](reports/model_comparison/model_comparison.html), and
[methodology/results](docs/model_comparison.md).

XGBoost + placeholder cleanup was selected narrowly on validation macro F0.5.
Its frozen evaluation macro F0.5 is **.989437**, link precision **.997631**, and
link recall **.972666**. Exact-set accuracy is .91. The small cleaning-related
validation improvement is not evidence of a general benefit. Evaluation is now
consumed, and must not be treated as fresh holdout data for further tuning.

From `lab_project/`, rebuild reports without retraining or rerunning evaluation:

```powershell
uv run python scripts/build_model_reports.py
uv run jupyter lab notebooks/05_model_comparison.ipynb
uv run python -m unittest discover -s scripts -p 'test_*.py' -v
node scripts/test_walkthrough.cjs
uv run python scripts/build_side_by_side.py
node scripts/test_side_by_side.cjs
```

Saved results, models, features, selection, and evaluation outputs are under
`artifacts/model_comparison_v1/`. Dependencies remain managed by uv and `uv.lock`.

## Cleaning audit and controlled comparison

See the executed [cleaning notebook](notebooks/03_cleaning_ablation.ipynb),
[interactive HTML report](reports/cleaning/cleaning_ablation.html), and
[methodology/results](docs/cleaning_ablation.md). At fixed K=10 we compared the
baseline, field-aware placeholder cleanup, and cleanup plus a small abbreviation
map. Every TF-IDF variant was fitted on training data only, with rules frozen
before validation access.

All three retain the same 5,182 of 5,189 validation true pairs (99.86510% recall).
Candidate counts are 55,430 / 55,426 / 55,425. No validation true pair is recovered
or newly lost. Both cleaned variants do recover and lose some training true pairs;
their net changes are not broad evidence of improvement. At that cleaning-only
stage, the baseline stayed adopted; the subsequent matching comparison selected
XGBoost with placeholder cleanup. All three variants remain preserved under
`artifacts/cleaning_v1/`. Legitimate Null/N/A names and Unicode non-joiners are
preserved. Raw data and `blocking_v1` candidate artifacts are unchanged.

From `lab_project/`:

```powershell
uv run jupyter lab notebooks/03_cleaning_ablation.ipynb
uv run python scripts/build_cleaning_report.py
```

The cleaning report contains before/after text examples, a candidate-rank explorer, audit
counts, recovered/newly lost pair counts, candidate churn, country/source slices,
and timings. That historical stage preceded classifier fitting and evaluation;
see the model section above for the current project state.

## Candidate generation

Open the executed [candidate-generation notebook](notebooks/02_candidate_generation.ipynb),
the [offline interactive report](reports/blocking/candidate_generation.html), or
the [findings](reports/blocking/findings.md). The notebook explains text normalization,
TF-IDF weights, the name/address union, coverage/cost trade-offs, missed pairs,
individual ranked candidates, and a small diagnostic graph.

On the reduced validation pool, exact-name blocking retained 22.18% of true pairs.
The name/address TF-IDF union retained 99.865% at K=10 (55,430 candidates), and
100% at K=25 (144,411 candidates) and K=50 (290,874 candidates). These are measured
blocking recalls, not classifier scores or guarantees on larger data. K is per
view and target source. The fixed minimum-cost/98%-recall rule selected K=10.

Saved outputs are in `artifacts/blocking_v1/`: candidate TSVs (including empty
reference rows), per-view rank/cosine evidence in Parquet, query diagnostics,
per-business counts, miss examples, metrics, and fitted training-only vectorizers.
`run.json` records versions, parameters, input/artifact hashes, timing, selection,
and training diagnostics. There are 260,063 training and 55,430 validation
candidates for the selected method. All seven missed validation pairs remain
absent from the candidate set. Training recall is an in-sample diagnostic.

The blocking experiment used only training and validation files. At that stage,
no classifier was trained and evaluation was unopened. Oracle F0.5 is explicitly an
upper bound, not an achieved final-model score.

From `lab_project/`:

```powershell
uv run jupyter lab notebooks/02_candidate_generation.ipynb
uv run python scripts/build_blocking_report.py
uv run python -m unittest discover -s scripts -p 'test_*.py' -v
```

The notebook reads saved experiment artifacts; restarting and running all cells
rebuilds its visualizations without silently changing the selected experiment.
`scripts/build_blocking_report.py` is its authoring source and overwrites its
generated notebook/report. To repeat the experiment into a new directory:

```powershell
uv run python scripts/run_blocking.py --output-dir artifacts/blocking_repeat
uv run python scripts/build_blocking_report.py --run-dir artifacts/blocking_repeat
```

The experiment refuses existing output directories; `INCOMPLETE.txt` marks an
interrupted/failed run. Timing excludes CSV/Parquet export; TF-IDF fit and
validation transforms are separate common setup costs. Name/address ablation
timings share a two-view retrieval run. Peak memory was not measured.

If Windows denies a uv cache rename during dependency installation, the serial
temporary-cache retry used successfully here was:

```powershell
$env:UV_CONCURRENT_DOWNLOADS = '1'
$env:UV_CONCURRENT_INSTALLS = '1'
uv sync --locked --no-cache --link-mode copy
```

## Training-data EDA

The executed [notebook](notebooks/01_training_eda.ipynb) uses Polars and Plotly.
Read the [offline interactive report](reports/training_eda/training_eda.html),
[findings](reports/training_eda/findings.md), or the full CSV tables in
`reports/training_eda/tables/`. Only the reduced **training** split is explored;
validation, evaluation, and competition test records remain reserved. This phase
contains descriptive analysis only, with no matching algorithm or model training.

From `lab_project/`, set up Python 3.13 using uv:

```powershell
uv venv --python 3.13
uv sync --locked
uv run jupyter lab notebooks/01_training_eda.ipynb
```

Choose the environment's Python kernel. The notebook includes inline interactive
charts and explains each analysis. If opening an already executed notebook, trust
it in Jupyter to allow its embedded Plotly JavaScript to run. Charts work offline.
Restart the kernel and run all cells in order for a clean interactive rerun.

To rebuild, execute, and export the notebook and report from `lab_project/`:

```powershell
uv run python scripts/build_eda.py
```

`build_eda.py` is the notebook authoring source: rebuilding replaces the notebook
and report. Put lasting changes there. The run metadata records versions, training
hashes, and chart/table inventory in `reports/training_eda/eda_run.json`.
`uv.lock` pins resolved dependencies; `.venv/` is local and should not be committed.

```text
lab_project/
├── data/dataset_small_10k/
│   ├── train/
│   ├── validation/
│   ├── evaluation/
│   ├── manifest.json
│   ├── README.md
│   └── VERIFICATION.md
├── notebooks/                 # executed EDA, blocking, cleaning, feature/model notebooks
├── reports/                   # offline HTML, findings, and CSV tables
├── artifacts/blocking_v1/     # candidate sets, evidence, and experiment metadata
├── artifacts/cleaning_v1/     # frozen cleaning audit and controlled comparisons
├── artifacts/model_comparison_v1/ # fitted models and frozen evaluation results
├── scripts/
│   ├── create_lab_subset.py
│   ├── test_create_lab_subset.py
│   ├── build_eda.py
│   ├── eda_helpers.py
│   ├── blocking.py
│   ├── er_metrics.py
│   ├── run_blocking.py
│   ├── build_blocking_report.py
│   ├── test_blocking.py
│   ├── cleaning.py
│   ├── run_cleaning.py
│   ├── build_cleaning_report.py
│   └── test_cleaning.py
├── pyproject.toml
└── uv.lock
```

See [dataset instructions](data/dataset_small_10k/README.md) and the
[verification report](data/dataset_small_10k/VERIFICATION.md).

From the repository root, regenerate into a new directory:

```powershell
python lab_project/scripts/create_lab_subset.py --output-dir lab_project/data/dataset_small_10k_repeat
```

The script reads the original competition training data by default. It never
overwrites an existing output directory. Default paths work from any working
directory; explicit relative paths are relative to your working directory.

Run the preparation checks from the repository root:

```powershell
python -m unittest discover -s lab_project/scripts -p test_create_lab_subset.py -v
```
