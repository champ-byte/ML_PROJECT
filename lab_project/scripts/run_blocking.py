"""Run the bounded blocking comparison. Never opens the evaluation split."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import platform
import time

import polars as pl
import sklearn

from blocking import TfidfBlocker, candidate_map, country_key, eligible_country, exact_candidates
from er_metrics import blocking_metrics


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data/dataset_small_10k"
DEFAULT_OUTPUT = PROJECT / "artifacts/blocking_v1"


def load_split(data, split, manifest):
    if split not in ("train", "validation"):
        raise ValueError("This experiment may read only train and validation")
    loaded, hashes = {}, {}
    for name in ("source1", "source2", "source3", "ground_truth"):
        path = data / split / f"{name}.tsv"
        hashes[f"{split}/{name}.tsv"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[f"{split}/{name}.tsv"] != manifest["tsv_sha256"][f"{split}/{name}.tsv"]:
            raise ValueError(f"Input hash changed: {path}")
        loaded[name] = pl.read_csv(path, separator="\t", infer_schema=False)
    truth = {record_id: set(value.split(",")) if value else set()
             for record_id, value in loaded.pop("ground_truth").iter_rows()}
    reference = set(loaded["source1"]["entity_id"])
    target = set(loaded["source2"]["entity_id"]) | set(loaded["source3"]["entity_id"])
    if reference != set(truth) or not all(value <= target for value in truth.values()):
        raise ValueError("Incomplete ground-truth membership")
    return loaded, truth, hashes


def write_candidates(folder, candidates, valid_targets):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for record_id in sorted(candidates):
            if not candidates[record_id] <= valid_targets:
                raise ValueError("Candidate not present in target files")
            writer.writerow([record_id, ",".join(sorted(candidates[record_id]))])
    with (folder / "candidate_pairs.tsv").open(encoding="utf-8", newline="") as stream:
        restored = {row["source1_entity_id"]: set(filter(None, row["candidate_entity_ids"].split(",")))
                    for row in csv.DictReader(stream, delimiter="\t")}
    if restored != candidates:
        raise ValueError("Candidate export round-trip failed")


def details(truth, candidates, source1):
    countries = dict(source1.select("entity_id", "country").iter_rows())
    return pl.DataFrame([{"source1_entity_id": key, "country": countries[key],
        "true_matches": len(truth[key]), "candidates": len(candidates[key]),
        "retained": len(truth[key] & candidates[key]), "missed": len(truth[key] - candidates[key])}
        for key in sorted(truth)])


def slice_metrics(truth, candidates, sources, method):
    result = []
    target_total = sources["source2"].height + sources["source3"].height
    for country in sorted(sources["source1"]["country"].unique().to_list()):
        ids = sources["source1"].filter(pl.col("country") == country)["entity_id"].to_list()
        result.append({"method": method, "slice_type": "country", "slice": country,
            **blocking_metrics({key: truth[key] for key in ids}, {key: candidates[key] for key in ids}, target_total)})
    for source, prefix in [("source2", "S2-"), ("source3", "S3-")]:
        result.append({"method": method, "slice_type": "target_source", "slice": source,
            **blocking_metrics({key: {item for item in value if item.startswith(prefix)} for key, value in truth.items()},
                               {key: {item for item in value if item.startswith(prefix)} for key, value in candidates.items()}, sources[source].height)})
    return result


def diagnose_misses(truth, candidates, prepared):
    reference = prepared.sources["source1"].to_dicts()
    refs = {row["entity_id"]: (i, row) for i, row in enumerate(reference)}
    targets = {}
    available = {}
    for source in ("source2", "source3"):
        records = prepared.sources[source].to_dicts()
        available[source] = {country_key(row["country"]) for row in records} - {""}
        targets.update({row["entity_id"]: (source, i, row) for i, row in enumerate(records)})
    result = []
    for key in sorted(truth):
        i, query = refs[key]
        for target_id in sorted(truth[key] - candidates[key]):
            source, j, target = targets[target_id]
            scores = {view: float(prepared.matrices["source1"][view][i].multiply(prepared.matrices[source][view][j]).sum())
                      for view in ("name", "address")}
            if not eligible_country(country_key(query["country"]), country_key(target["country"]), available[source]):
                reason = "country_partition"
            elif all(prepared.matrices["source1"][view][i].nnz == 0 for view in ("name", "address")):
                reason = "zero_query_vectors"
            elif max(scores.values()) <= 0:
                reason = "no_shared_ngrams"
            else:
                reason = "outside_top_k"
            result.append({"source1_entity_id": key, "target_id": target_id, "target_source": source,
                "country": query["country"], "reason": reason, "name_cosine": scores["name"],
                "address_cosine": scores["address"], "reference_name": query["business_name"],
                "target_name": target["business_name"], "reference_address": query["business_address"],
                "target_address": target["business_address"]})
    schema = {"source1_entity_id": pl.String, "target_id": pl.String, "target_source": pl.String,
              "country": pl.String, "reason": pl.String, "name_cosine": pl.Float64,
              "address_cosine": pl.Float64, "reference_name": pl.String, "target_name": pl.String,
              "reference_address": pl.String, "target_address": pl.String}
    return pl.DataFrame(result, schema=schema)


def run(output_dir=DEFAULT_OUTPUT, data=DATA):
    output_dir, data = Path(output_dir).resolve(), Path(data).resolve()
    if output_dir.exists():
        raise ValueError(f"Output exists; use a new --output-dir: {output_dir}")
    output_dir.mkdir(parents=True)
    marker = output_dir / "INCOMPLETE.txt"
    marker.write_text("Experiment incomplete; do not use these outputs.\n", encoding="utf-8")
    started = time.perf_counter()
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    try:
        training, training_truth, train_hashes = load_split(data, "train", manifest)
        validation, truth, validation_hashes = load_split(data, "validation", manifest)
        assert set(training_truth).isdisjoint(truth)
        assert {item for frame in training.values() for item in frame["entity_id"]}.isdisjoint(
            {item for frame in validation.values() for item in frame["entity_id"]})
        target_count = validation["source2"].height + validation["source3"].height
        valid_targets = set(validation["source2"]["entity_id"]) | set(validation["source3"]["entity_id"])
        reference_ids = sorted(truth)
        rows, slices = [], []
        fit_start = time.perf_counter()
        blocker = TfidfBlocker(batch_size=64).fit(training)
        fit_seconds = time.perf_counter() - fit_start
        prep_start = time.perf_counter()
        prepared = blocker.prepare(validation)
        prepare_seconds = time.perf_counter() - prep_start
        print(f"Fitted training-only TF-IDF in {fit_seconds:.2f}s; prepared validation in {prepare_seconds:.2f}s", flush=True)
        exact_start = time.perf_counter()
        exact = exact_candidates(validation)
        exact_seconds = time.perf_counter() - exact_start
        rows.append({"method": "exact_name", "k": 0, "view": "exact", "retrieval_seconds": exact_seconds,
                     "runtime_scope": "exact index build and lookup", **blocking_metrics(truth, exact, target_count)})
        write_candidates(output_dir / "validation/exact_name", exact, valid_targets)
        details(truth, exact, validation["source1"]).write_csv(output_dir / "validation/exact_name/per_business.csv")
        slices.extend(slice_metrics(truth, exact, validation, "exact_name"))
        candidates_by_method = {"exact_name": exact}
        for k in (10, 25, 50):
            retrieval_start = time.perf_counter()
            evidence, diagnostics = blocker.retrieve(prepared, k)
            elapsed = time.perf_counter() - retrieval_start
            for view in ("name", "address", "union"):
                method = f"tfidf_{view}_k{k}"
                candidates = candidate_map(reference_ids, evidence, None if view == "union" else view)
                summary = blocking_metrics(truth, candidates, target_count)
                rows.append({"method": method, "k": k, "view": view, "retrieval_seconds": elapsed,
                             "runtime_scope": "both TF-IDF views; shared by view ablations", **summary})
                if view == "union":
                    folder = output_dir / "validation" / method
                    write_candidates(folder, candidates, valid_targets)
                    evidence.write_parquet(folder / "retrieval_evidence.parquet")
                    diagnostics.write_csv(folder / "query_diagnostics.csv")
                    details(truth, candidates, validation["source1"]).write_csv(folder / "per_business.csv")
                    slices.extend(slice_metrics(truth, candidates, validation, method))
                    candidates_by_method[method] = candidates
                    print(f"{method}: recall={summary['pair_recall']:.4%}, pairs={summary['candidate_pairs']:,}, retrieval={elapsed:.2f}s", flush=True)
        metrics = pl.DataFrame(rows)
        metrics.write_csv(output_dir / "validation_metrics.csv")
        pl.DataFrame(slices).write_csv(output_dir / "validation_slices.csv")
        eligible = [row for row in rows if row["view"] in ("union", "exact") and row["pair_recall"] >= .98]
        selected = min(eligible, key=lambda row: (row["candidate_pairs"], row["retrieval_seconds"], row["method"])) if eligible else None
        diagnostic = selected if selected and selected["view"] == "union" else max(
            (row for row in rows if row["view"] == "union"), key=lambda row: row["pair_recall"])
        missed = diagnose_misses(truth, candidates_by_method[diagnostic["method"]], prepared)
        missed.write_csv(output_dir / "missed_true_pairs.csv")
        training_summary = None
        if selected:
            print(f"Selected {selected['method']} by validation recall/cost. Generating training candidates...", flush=True)
            train_start = time.perf_counter()
            if selected["method"] == "exact_name":
                train_candidates = exact_candidates(training)
            else:
                train_prepared = blocker.prepare(training)
                train_evidence, train_diagnostics = blocker.retrieve(train_prepared, selected["k"])
                train_candidates = candidate_map(sorted(training_truth), train_evidence)
            train_targets = set(training["source2"]["entity_id"]) | set(training["source3"]["entity_id"])
            folder = output_dir / "train" / selected["method"]
            write_candidates(folder, train_candidates, train_targets)
            if selected["method"] != "exact_name":
                train_evidence.write_parquet(folder / "retrieval_evidence.parquet")
                train_diagnostics.write_csv(folder / "query_diagnostics.csv")
            training_summary = {**blocking_metrics(training_truth, train_candidates, len(train_targets)),
                                "preparation_and_retrieval_seconds": time.perf_counter() - train_start,
                                "scope": "in-sample training diagnostic, not held-out performance"}
        else:
            print("No method meets 98% validation recall. Stopping at blocking diagnostics; no training candidates or classifier generated.", flush=True)
        with (output_dir / "tfidf_vectorizers.pkl").open("wb") as stream:
            pickle.dump(blocker.vectorizers, stream)
        result = {"status": "complete", "classifier_trained": False, "evaluation_opened": False,
            "protocol": {"ks": [10,25,50], "recall_target": .98, "ngram_range": [3,5],
                         "analyzer": "char", "batch_size": 64, "dtype": "float32",
                         "fit_scope": "all training source records only", "country_fallback": "blank or absent query label searches target source globally; blank target labels always eligible"},
            "versions": {"python": platform.python_version(), "polars": pl.__version__, "sklearn": sklearn.__version__},
            "input_hashes": {**train_hashes, **validation_hashes},
            "script_hashes": {name: hashlib.sha256((PROJECT / 'scripts' / name).read_bytes()).hexdigest()
                              for name in ('blocking.py','er_metrics.py','run_blocking.py')},
            "fit_seconds": fit_seconds, "validation_prepare_seconds": prepare_seconds,
            "vocabulary_sizes": {view: len(vectorizer.vocabulary_) for view, vectorizer in blocker.vectorizers.items()},
            "selected_method": selected["method"] if selected else None,
            "diagnostic_method": diagnostic["method"], "diagnostic_k": diagnostic["k"],
            "recall_target_met": bool(selected), "training_summary": training_summary,
            "elapsed_seconds": time.perf_counter() - started,
            "memory_note": "Peak memory not measured; sparse multiplication is bounded by batch_size times eligible target count.",
            "artifact_sha256": {path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(output_dir.rglob('*')) if path.is_file() and path.name != 'INCOMPLETE.txt'}}
        (output_dir / "run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        marker.unlink()
        return result
    except Exception as error:
        marker.write_text(f"Experiment failed: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output_dir)
