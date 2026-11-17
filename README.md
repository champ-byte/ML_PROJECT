# Business Entity Resolution — ML Lab

Experiments in business-record matching using candidate generation, pairwise
similarity features, logistic regression, and XGBoost, with interactive explanations.

- `lab_project/`: lab source code, uv configuration/lockfile, methodology, and references.
- `student_resource/`: expected local location for the separately supplied competition package.

See [lab instructions](lab_project/README.md) and
[methodology and results](lab_project/docs/model_comparison.md).

## Local setup

Use Python 3.13 and uv:

```powershell
cd lab_project
uv venv --python 3.13
uv sync --locked
```

Competition datasets are not included in Git. Place the original training/test
TSVs under `student_resource/dataset/train/` and `student_resource/dataset/test/`
as described in the competition README. Follow the lab instructions to regenerate
the reduced dataset and experiment artifacts before building the notebooks/reports.

Generated datasets, fitted models, executed notebooks, HTML reports, environments,
and caches stay local. Their preparation and notebook/report authoring scripts
are tracked in `lab_project/scripts/`. No GitHub remote is configured automatically.
