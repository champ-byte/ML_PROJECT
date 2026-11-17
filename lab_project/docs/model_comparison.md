# Pair features, matching models, and interactive walkthrough

## Completed experiment

The fixed K=10 blocker now feeds two binary classifiers across three preprocessing
variants. All six pipelines were trained sequentially on their saved training
candidates, then compared on validation. One winner and threshold were frozen
before any evaluation records were opened. The selected model was not refitted.

- [Interactive walkthrough](../reports/model_comparison/interactive_walkthrough.html)
- [Side-by-side learned weights and explanations](../reports/model_comparison/side_by_side_comparison.html)
- [Executed feature notebook](../notebooks/04_pair_features.ipynb)
- [Executed comparison notebook](../notebooks/05_model_comparison.ipynb)
- [Feature report](../reports/features/features.html)
- [Model comparison report](../reports/model_comparison/model_comparison.html)

The walkthrough uses recorded validation examples, including deliberately selected
errors. It is not an arbitrary-text inference service. Choose a pipeline/business,
step through records → bucketing/TF-IDF → similarities → learned weights → decision, inspect a candidate's
score contributions, and move the validation threshold. Metric cards cover all
1,500 validation businesses, not just displayed examples. The evaluation readout
is fixed and cannot be tuned using those controls. All JavaScript and chart assets
are embedded for offline use.

The side-by-side page keeps the reference/candidate pair identical for both models.
Its first tab shows scores, selected thresholds, decisions, and additive raw-margin
explanations: logistic intercept plus standardized feature × coefficient, versus
XGBoost's SHAP expected-margin term plus per-feature SHAP contributions. These
different base references mean contributions should be read as model-specific
explanations, rather than equivalent parameters or causal effects. Saved scores
determine decisions; displayed arithmetic is rounded.

The second tab compares logistic coefficients across all three preprocessing
variants, with a raw-unit slope option (coefficient / training scale). It also
compares mean absolute contributions for both models over the **50,501 validation
candidate pairs retrieved by all three variants**, so the contribution population
stays fixed. Full validation accuracy metrics still cover each pipeline's full
candidate set and all businesses. The displayed reference examples are deliberately selected diagnostics, including
disagreements chosen deterministically from saved validation predictions. A pair missing from the selected variant has no
score or explanation there. No model training or evaluation inference is performed.

Rebuild this comparison alone with `uv run python scripts/build_side_by_side.py`
and check its controls with `node scripts/test_side_by_side.cjs` from `lab_project/`.
The full report builder also rebuilds and links it. The JSON data and manifest
alongside the HTML record its source scope and common-cohort checksum.

## Features and fit protocol

Twenty features, ten each for name/address: TF-IDF cosine, token Jaccard,
normalized Levenshtein similarity, length ratio, exact nonblank equality,
digit-token Jaccard, digit-token conflict, missing left/right, and token containment.
Features stay in [0,1]. Missing similarities are zero, with explicit missingness
flags; empty/empty is not an exact match. Digit conflict requires two present but
unequal digit sets. Leading zeros are retained.

IDs, ground truth, source labels, country one-hot labels, and manifest distractor
membership are excluded. Pair features use each variant's own frozen text rules
and training-fitted vectorizers; no new TF-IDF fit occurs on validation/evaluation.
All retrieved negatives are used. Unretrieved positives are not injected into
training or candidate sets. Features are computed in bounded sparse batches.

- Logistic regression: training-only StandardScaler, C=1, lbfgs, max_iter=1000.
- XGBoost: binary logistic objective, 200 trees, depth 3, learning rate .05,
  min_child_weight 5, lambda 1, histogram tree method, two CPU threads.
- Both: seed 42; no class weighting, subsampling, early stopping, or broad search.

Each threshold sweep processes every distinct validation score, accepting tied
scores together. Select maximum per-business macro F0.5, then precision, then
higher threshold. Pipeline selection uses validation macro F0.5, then precision,
then simpler model/variant tie-breaks. The protocol and feature contract were
written before fitting. Evaluation selection was written and hash-checked before
opening evaluation sources; predictions were saved before opening their labels.

## Validation results

| Text/candidate variant | Model | Macro F0.5 | Link precision | Link recall |
|---|---|---:|---:|---:|
| Baseline | Logistic regression | .987081 | .993522 | .975332 |
| Baseline | XGBoost | .990387 | .995495 | .979572 |
| Placeholder cleanup | Logistic regression | .987314 | .996436 | .969936 |
| Placeholder cleanup | XGBoost | **.990481** | .995303 | .980150 |
| Cleanup + abbreviations | Logistic regression | .986722 | .994296 | .974176 |
| Cleanup + abbreviations | XGBoost | .989170 | .994516 | .978609 |

Selected: **XGBoost + placeholder cleanup**, threshold **0.4353279173374176**.
Its macro F0.5 improvement over baseline XGBoost is only .000094235 (about .0094
percentage points). This narrow validation win is not strong evidence of a broad
cleaning benefit. Abbreviation expansion reduced validation scores for both models
in this fixed first comparison. No significance claim or further tuning is made.

Cleaning affects both candidate identities and features, so these are pipeline
comparisons, not isolated causal effects of feature cleaning alone.

## One frozen evaluation pass

| Measure | Result |
|---|---:|
| Macro F0.5 | .989437 |
| Link precision | .997631 |
| Link recall | .972666 |
| Link F1 | .984990 |
| Exact-set accuracy | .910000 |
| Correct empty predictions on true singletons | 83 / 83 |
| True positive / false positive / false negative links | 5,053 / 12 / 142 |

Of 142 missed links, 12 were lost during blocking and 130 rejected by the
classifier. Correct singleton predictions earn 1 in per-business macro F0.5.
Metrics include every evaluation reference business; blocked positives remain FN.
Candidate-conditional average precision is a secondary diagnostic, not a substitute
for end-to-end scores. Per-country, match-count, and target-source link slices are
exported. Source-restricted singleton/macro scores are not treated as official
competition scores.

The 95% bootstrap interval for evaluation macro F0.5 is [.986193, .992234], using
2,000 resamples of Source 1 businesses with seed 42. It conditions on this fixed
pipeline and excludes training/selection uncertainty. Zero observed singleton
false merges among 83 cases does not establish a zero population false-merge rate.

This evaluation split is now **consumed**. Future experiments should compare on
training/validation and use a genuinely fresh holdout for an independent final
estimate. Full competition-pool scaling and unseen-France performance remain
unmeasured. No official competition test data or external identity services were used.

## Explanations and interactive controls

Logistic explanations are standardized values × coefficients plus the intercept.
XGBoost uses native SHAP contributions plus its base term. Every saved contribution
row was checked to reconstruct the raw model margin. Contributions explain
log-odds, not causal effects or percentage-point changes in calibrated probabilities.

The bucketing step shows the actual country-eligible pool, separate name/address
top-10 lists for each target source, overlap, and the deduplicated union. A worked
fragment table shows raw counts, training document frequencies (recovered from
the saved smoothed IDF), IDF, L2-normalized weights, and cosine contributions.
Remaining contributions are included in the full total. No validation IDF is fitted.

The weights step shows actual logistic training mean/scale, coefficients, intercept,
and optimizer iteration count, then works through standardized values → weighted
sum → sigmoid. Individual L-BFGS updates were not logged; no fictitious trajectory
is displayed. For XGBoost, inspect actual split paths and leaf outputs at rounds
1, 25, 100, and 200. Gradient/Hessian sums and training losses are reconstructed
from frozen tree prefixes on training candidates. The binary logistic formulas are
`g=score-label`, `h=score*(1-score)`, and `leaf=-sum(g)/(sum(h)+lambda)`.
Stored leaf outputs already contain learning-rate shrinkage and are added once.
The sum of all 200 outputs plus initial margin reconstructs the score. That initial
boosting margin differs from SHAP's expected-margin base reference. No new model
fit or evaluation inference is performed to build these explanations.

The walkthrough's threshold grid has 0.005 steps plus each model's exact selected
threshold. Every displayed metric is exact for that shown threshold; no metric
interpolation is used. Pair features and scores preserve full precision; displayed
labels are rounded, and SHAP display contributions retain seven decimals. Exported
weight arithmetic can differ from native float32 accumulation in the final digits;
actual decisions always use saved full-precision scores. Ground truth can be toggled as a post-hoc overlay and
never supplies inference features or inserts missed candidates.

Python tests check manual metrics, tied thresholds, blocked FNs, nonblank equality,
missingness, TF-IDF calculations, Newton leaf weights, and explanation additivity.
Pure Node/DOM-stub tests exercise 342 step/threshold/weight states across all pipelines/examples and check that evaluation
readouts remain fixed. They are functional checks, not browser layout screenshots.

## Reproduce without rerunning evaluation

From `lab_project/`, use the existing uv environment/lockfile:

```powershell
uv sync --locked
uv run jupyter lab notebooks/04_pair_features.ipynb notebooks/05_model_comparison.ipynb
uv run python scripts/build_model_reports.py
uv run python -m unittest discover -s scripts -p 'test_*.py' -v
node scripts/test_walkthrough.cjs
```

`build_model_reports.py` is the notebook/report authoring source. It regenerates
reports and the interactive HTML from saved artifacts, without training or running
evaluation again. `walkthrough.html` is the editable HTML template;
`walkthrough_learning.js` defines the new views, and `walkthrough_weights.py`
reconstructs their data. To rebuild only the walkthrough:

```powershell
uv run python -c "import sys; sys.path.insert(0, 'scripts'); from build_model_reports import walkthrough; walkthrough()"
```

Final artifacts
live under `artifacts/model_comparison_v1/`: features, fitted models, all validation
curves/predictions/errors/contributions, selection/protocol, evaluation candidates,
predictions, matching TSV, metrics, hashes, and versions. XGBoost also has a native
`.ubj` model export. The run refuses an existing output directory. Do not use new
output directories as a way to tune repeatedly against the consumed evaluation.

## References

The shared [BibTeX bibliography](references.bib) includes the blocking, XGBoost,
and feature-attribution papers. Primary method/implementation references:

- Chen and Guestrin (2016), [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).
- Lundberg, Erion, and Lee (2018), [Consistent Individualized Feature Attribution for Tree Ensembles](https://arxiv.org/abs/1802.03888).
- Papadakis et al. (2020), [A Survey of Blocking and Filtering Techniques for Entity Resolution](https://arxiv.org/abs/1905.06167).
- [scikit-learn logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- [RapidFuzz normalized Levenshtein similarity](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html).
- [XGBoost prediction and contribution API](https://xgboost.readthedocs.io/en/stable/prediction.html).
- [Boosted-tree leaf-weight derivation](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).
- [Smoothed TF-IDF and L2 normalization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html).

These references motivate the methods; all performance numbers above are from
this local experiment and are not supplied by those papers.
