# Algorithm selection, evaluation, and visualization

Status: candidate generation, cleaning, and matching models implemented and measured,
2026-11-16. The design below records the original proposal. See the
[model comparison](model_comparison.md) for the six-pipeline comparison and completed
single frozen evaluation pass; evaluation is now consumed.
See [blocking findings](../reports/blocking/findings.md) for current evidence.
Validation union recall was 99.865% at K=10 and 100% at K=25/50; the predefined
minimum-cost/98%-recall rule selected K=10. These finite reduced-pool results do
not establish full-pool performance. Bibliography: [references.bib](references.bib).

A subsequent [fixed-K cleaning ablation](cleaning_ablation.md) found unchanged
validation true-pair coverage for conservative placeholder/abbreviation variants.
The original baseline remains adopted; cleaning branches are preserved for later
model comparisons. Those comparisons are now completed in the model report above.

## Evidence and proposed algorithms

The training EDA contains 7,000 reference businesses, 24,225 labeled match pairs,
and 7,000 additional target records. There are 1,247 missing addresses across the
38,225 training records. Of the labeled pairs, 21,696 have different names even
after lowercasing and trimming. These differences motivate approximate matching;
they do not establish which approximate algorithm will work best.

An additional training-only check found zero country disagreements among all
24,225 labeled pairs. This supports using country as an initial partition, but
does not prove that cross-country matches never occur in other data.

| Phase | Baseline | Recommended candidate |
|---|---|---|
| Candidate generation (blocking / bucketing) | Same country + exact normalized full name | Union of name and address character-ngram TF-IDF top-K retrieval |
| Pairwise match decision | Standardized similarity features + logistic regression | Same feature set + XGBoost |

The blocking literature distinguishes signature-based blocks from nearest-neighbor
retrieval [1,2]. Our recommended candidate generator is sparse retrieval; it does
not assign each business to one exclusive cluster. Overlapping candidates are
appropriate because one reference business can have many matching records.

Use logistic regression as an interpretable benchmark, and XGBoost as the main
nonlinear candidate [3]. Select the final model using validation macro F0.5; there
is no evidence yet that XGBoost will outperform logistic regression on this data.
XGBoost is Apache-2.0 licensed [8], consistent with the competition's stated
final-model license requirement. The competition README remains authoritative.

## Phase 1: generate candidates

1. Preserve raw fields. Create derived Unicode-normalized, lowercased text with
   consistent whitespace/punctuation. Retain non-Latin characters and accents;
   do not use external business lookups, address services, or data augmentation.
2. Create separate name and address TF-IDF views using character ngrams of lengths
   3–5. Fit vocabulary and IDF on training records only; freeze for later splits.
   TF-IDF configuration and character analyzers are documented in [6].
3. Search Source 2 and Source 3 separately within dynamically discovered country
   partitions. Do not hard-code US/India or discard unseen countries such as France.
   Missing/unknown country gets an explicit fallback search, not silent exclusion.
4. Retrieve top-K positive-cosine candidates per view, per target source, then
   take their deduplicated union. Skip the address view for blank addresses and
   skip zero-vector queries rather than returning arbitrary tied records. Record
   zero-vector/no-candidate cases; they need investigation, especially for unseen
   scripts/countries. An entirely out-of-vocabulary record may need a later
   token-based fallback; do not claim the initial design resolves that failure.
5. Compare K = 10, 25, 50 per view/source. With two views and two target sources,
   the upper bound is 4K distinct candidates per reference business, before overlap.
   Do not cap the union again without measuring the extra recall loss.
6. Use sparse, batched retrieval; avoid materializing a dense all-pairs matrix.
   This is a lab-scale experiment. Full competition-scale retrieval requires a
   separate runtime/memory assessment.

Proposed validation target: at least 98% pair recall, with a preference for 99%
if candidate cost permits. Choose the lowest-cost tested setting meeting 98%,
using candidate count and then elapsed time as tie-breakers. If none meets it,
report the shortfall and investigate misses before classifier tuning. Exact-name
blocking is the reference comparison, not a presumed adequate solution.

Missing addresses and transliterations across scripts can defeat text overlap.
Character ngrams help with local spelling variation but are not a transliteration
model. Audit these cases explicitly. Also report any validation true pairs lost
solely to country partitioning.

Save the exact candidate set fed to inference, with retrieval view, rank, cosine
score, and query/target IDs. Ground truth is used for evaluation, never to insert
otherwise unretrieved pairs into validation/evaluation candidates.

## Phase 2: score candidates and return sets

Start with a small shared feature set: name/address TF-IDF cosine, token Jaccard,
normalized edit similarity, length ratios, exact normalized agreement, digit-token
agreement/conflict, and missing-field indicators. A missing address is not an exact
address match. Do not use record IDs or the manifest's distractor membership as
predictive features. Avoid training-country one-hot encodings that assume a closed set.

Train both classifiers on the same training candidates and labels, including the
confusing nonmatches actually retrieved. Start with all retrieved negatives. If
negative subsampling later becomes necessary, record its rate and apply it only
to training; validation/evaluation retain their full candidate sets. Never create
a random pair split that puts the same business in training and validation.

Each candidate gets a score. Accept all candidates at or above a validation-chosen
threshold; return an empty list if none passes. Do not force a single best match,
one-to-one assignment, or a fixed number of answers. Raw classifier scores are
not assumed calibrated probabilities.

Use a bounded first comparison: fixed blocker, shared features, one regularized
logistic model and one shallow XGBoost configuration; choose a separate threshold
for each on validation macro F0.5. Avoid a broad search until errors identify a
specific limitation. Freeze preprocessing, features, blocker, model, and threshold
before evaluating the held-out evaluation split. Retain the train-fitted model
for this first evaluation so the selected threshold is not transferred to a
newly refitted model without validation.

## Metrics and their scope

Let T_i be the true target-ID set for reference business i, C_i its candidates,
and M_i the accepted matches. Always enforce M_i as a subset of C_i.

| Metric | Definition / purpose |
|---|---|
| Blocking pair recall (pair completeness) | sum_i size(C_i intersect T_i) / sum_i size(T_i) |
| Complete-match coverage | Fraction of nonsingleton reference businesses for which all true matches survive blocking |
| Blocking pair quality | Retained true pairs / total candidate pairs; diagnostic, not the classifier's precision |
| Reduction ratio | 1 - candidate pairs / [N1 * (N2 + N3)]; denominator is the full split's cross-source comparison space, before country filtering |
| Candidate workload | Total, mean, median, p95 candidates per reference business, plus zero-candidate rate |
| Primary: macro F0.5 | Compute the competition score per reference business and average across all businesses, including singletons |
| End-to-end micro precision / recall / F1 | Pool TP, FP, FN over all reference businesses; expose the total link trade-off |
| Exact-set accuracy | Fraction with M_i exactly equal to T_i |
| Singleton correctness | Fraction of true singletons assigned an empty list; also report the complementary false-merge rate |
| Candidate-conditional AP | Average precision from candidate scores only; label this scope and report blocking recall alongside it [7] |
| Cost | Candidate-generation, feature, fit and inference time; peak memory where measurable |
| Slices | Country, target source, missing address, and true match count; show denominators |

For each business TP_i = size(M_i intersect T_i), FP_i = size(M_i minus T_i),
FN_i = size(T_i minus M_i). Its score is:

```text
F0.5_i = 1.25 * TP_i / (1.25 * TP_i + FP_i + 0.25 * FN_i)
```

When both the true and predicted sets are empty, assign 1 as required by the
competition. All other zero-overlap cases score 0. Average these per-business
scores; do not calculate F0.5 from averaged precision/recall. For auxiliary micro
precision with no predictions, report 0 with a no-predictions flag; singleton
credit belongs in the macro competition score.

Count unretrieved true matches as end-to-end false negatives. A high score on
retrieved candidates cannot compensate for a blocker that removed true matches.
Compute an oracle blocking score by predicting C_i intersect T_i; this shows the
highest macro F0.5 attainable with perfect classification of the retained pairs.

Do not headline ordinary pair accuracy: the overwhelmingly large number of
nonmatches can make it misleading. PR curves/AP focus on positive predictions
[5], but are population-dependent and not substitutes for the competition score.
Candidate-only confusion matrices exclude unretrieved positive pairs; display
that omission separately. Use a TP/FP/FN summary for end-to-end outcomes.

After the pipeline is frozen, add a 95% bootstrap interval for evaluation macro
F0.5 by resampling Source 1 businesses (not individual pairs), with a recorded seed.
Evaluation results describe this reduced pool; they are not leaderboard estimates
or evidence of performance on France, which is absent from the labeled subset.

## Visualization design

Stay with Jupyter + Polars + Plotly and offline HTML exports.

| View | What the student can inspect |
|---|---|
| Candidate explorer | One reference record, ranked candidates from both sources, contributing retrieval views, and post-hoc truth labels |
| Blocking trade-off | Recall vs candidate count/runtime for K=10/25/50; exact-name baseline and proposed target marked |
| Retrieval coverage | Candidate-count histogram and miss-rate slices; incremental matches recovered by the name/address union |
| Small bipartite graph | Reference businesses on one side, target records on the other; candidate/accepted edges distinguished. Any unretrieved truth edges are explicitly post-hoc diagnostic overlays |
| Score distributions | True and false candidate pairs, with the selected decision threshold |
| Validation threshold explorer | Linked micro precision, micro recall, macro F0.5, singleton correctness and exact-set accuracy as the threshold changes |
| Decision explanation | Similarity features side by side; logistic feature contributions or XGBoost SHAP contributions for individual pairs |
| Error decomposition | True links recovered, lost in blocking, and rejected by the classifier; false accepted links counted separately |
| Frozen evaluation summary | Per-model comparison and country/source slices, without interactive threshold tuning on evaluation labels |

XGBoost's contribution API supports local explanations [4,9]. Label contribution
axes in raw margin/log-odds units where applicable; they explain the score, not
causal effects or percentage-point changes in a calibrated match probability.

## Implementation order

1. Implement and test competition-compatible metrics on hand-computed examples.
2. Implement both blockers and their validation coverage/cost visualizations.
3. Freeze the blocker, construct shared candidate features, compare both classifiers.
4. Select thresholds on validation and build pair explanations/error views.
5. Freeze the pipeline, run evaluation once, and export reports and final match sets.

Use uv for dependencies and continue to keep all lab artifacts outside
`student_resource/`. Keep candidate and matching outputs compatible with the
competition's TSV format when producing submission-style artifacts.

## References

1. Papadakis, Skoutas, Thanos, and Palpanas. *A Survey of Blocking and Filtering
   Techniques for Entity Resolution*. arXiv:1905.06167, revised 2020.
   [Paper](https://arxiv.org/abs/1905.06167). Supports the stage separation and candidate-quality metrics.
2. Papadakis et al. *How to reduce the search space of Entity Resolution: with
   Blocking or Nearest Neighbor search?* 2022.
   [Paper](https://arxiv.org/abs/2202.12521). Supports comparing candidate-selection approaches and configurations; does not establish the winner on our data.
3. Chen and Guestrin. *XGBoost: A Scalable Tree Boosting System*. KDD 2016.
   [Paper](https://arxiv.org/abs/1603.02754). Foundation for the proposed nonlinear classifier.
4. Lundberg, Erion, and Lee. *Consistent Individualized Feature Attribution for
   Tree Ensembles*. 2018. [Paper](https://arxiv.org/abs/1802.03888).
   Foundation for tree-model feature attribution.
5. Saito and Rehmsmeier. *The Precision-Recall Plot Is More Informative than the
   ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE,
   2015. [Paper](https://doi.org/10.1371/journal.pone.0118432).
6. [TF-IDF API documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
7. [Average precision API documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
8. [XGBoost license](https://github.com/dmlc/xgboost/blob/master/LICENSE).
9. [XGBoost prediction and contribution documentation](https://xgboost.readthedocs.io/en/stable/prediction.html).

These sources motivate methods and reporting choices. None supplies performance
results for this competition dataset. Research reading does not involve querying
external business identity information, which the competition prohibits.
