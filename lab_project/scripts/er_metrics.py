"""Set-based entity-resolution metrics, including the competition's singleton rule."""

import numpy as np


def f05(true_ids, predicted_ids):
    true_ids, predicted_ids = set(true_ids), set(predicted_ids)
    if not true_ids and not predicted_ids:
        return 1.0
    tp = len(true_ids & predicted_ids)
    return 1.25 * tp / (1.25 * tp + len(predicted_ids - true_ids) + .25 * len(true_ids - predicted_ids))


def set_metrics(truth, predictions):
    if not truth or set(truth) != set(predictions):
        raise ValueError("Predictions must include exactly one set for every reference business")
    tp = fp = fn = exact = singleton_correct = singleton_count = 0
    scores = []
    for record_id, expected in truth.items():
        expected, predicted = set(expected), set(predictions[record_id])
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        exact += expected == predicted
        if not expected:
            singleton_count += 1
            singleton_correct += not predicted
        scores.append(f05(expected, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "macro_f05": float(np.mean(scores)),
            "micro_precision": precision, "micro_recall": recall,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "no_predictions": tp + fp == 0, "exact_set_accuracy": exact / len(truth),
            "singleton_count": singleton_count,
            "singleton_correctness": singleton_correct / singleton_count if singleton_count else None}


def blocking_metrics(truth, candidates, target_count):
    if set(truth) != set(candidates) or not truth:
        raise ValueError("Candidate sets must include every reference business")
    total_true = sum(len(value) for value in truth.values())
    retained = sum(len(set(value) & set(candidates[key])) for key, value in truth.items())
    counts = np.array([len(candidates[key]) for key in truth])
    total_candidates = int(counts.sum())
    nonsingletons = [key for key, value in truth.items() if value]
    oracle = {key: set(value) & set(candidates[key]) for key, value in truth.items()}
    full_space = len(truth) * target_count
    return {"reference_businesses": len(truth), "true_pairs": total_true,
            "retained_true_pairs": retained, "missed_true_pairs": total_true - retained,
            "candidate_pairs": total_candidates,
            "pair_recall": retained / total_true if total_true else None,
            "complete_match_coverage": sum(set(truth[key]) <= set(candidates[key]) for key in nonsingletons) / len(nonsingletons) if nonsingletons else None,
            "pair_quality": retained / total_candidates if total_candidates else 0.0,
            "reduction_ratio": 1 - total_candidates / full_space if full_space else None,
            "full_comparison_space": full_space,
            "mean_candidates": float(counts.mean()), "median_candidates": float(np.median(counts)),
            "p95_candidates": float(np.quantile(counts, .95)), "max_candidates": int(counts.max()),
            "zero_candidate_rate": float(np.mean(counts == 0)),
            "oracle_macro_f05": set_metrics(truth, oracle)["macro_f05"]}
