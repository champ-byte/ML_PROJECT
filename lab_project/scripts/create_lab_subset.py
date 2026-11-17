"""Create a reproducible, business-level entity-resolution lab subset (stdlib only)."""

import argparse
import collections
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import tempfile
import time


SOURCE_HEADER = ["entity_id", "business_name", "business_address", "country"]
TRUTH_HEADER = ["source1_entity_id", "matched_entity_ids"]
SPLITS = ("train", "validation", "evaluation")
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR.parent / "student_resource" / "dataset" / "train"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "dataset_small_10k"


def rows(path, header):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        if next(reader, None) != header:
            raise ValueError(f"Unexpected header: {path}")
        for number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Malformed row {number}: {path}")
            yield row


def targets(value):
    result = value.split(",") if value else []
    if len(result) != len(set(result)) or any(
        not item.startswith(("S2-", "S3-")) for item in result
    ):
        raise ValueError(f"Invalid or duplicate ground-truth targets: {value}")
    return result


def allocate(counts, total):
    """Largest-remainder allocation with stable ties and integer arithmetic."""
    available = sum(counts.values())
    if total < 0 or total > available:
        raise ValueError(f"Requested {total} records from {available} available")
    if not available:
        return {key: 0 for key in sorted(counts)}
    result = {key: count * total // available for key, count in sorted(counts.items())}
    ranking = sorted(counts, key=lambda key: (-(counts[key] * total % available), key))
    for key in ranking[:total - sum(result.values())]:
        result[key] += 1
    return result


def partition(groups, rng):
    """Exact global 70/15/15 totals, with proportional stratum allocation."""
    remaining = {key: sorted(values) for key, values in sorted(groups.items())}
    for values in remaining.values():
        rng.shuffle(values)
    n = sum(map(len, remaining.values()))
    # Allocate percentages without restricting dataset size to the sum of weights.
    totals = {name: n * weight // 100 for name, weight in zip(SPLITS, (70, 15, 15))}
    ranking = sorted(zip(SPLITS, (70, 15, 15)), key=lambda item: (-(n * item[1] % 100), SPLITS.index(item[0])))
    for name, _ in ranking[:n - sum(totals.values())]:
        totals[name] += 1
    assigned = {}
    for split in SPLITS:
        quotas = allocate({key: len(value) for key, value in remaining.items()}, totals[split])
        for key, quota in quotas.items():
            for record_id in remaining[key][:quota]:
                assigned[record_id] = split
            remaining[key] = remaining[key][quota:]
    return assigned, totals


def reservoir(pools, seen, key, item, quota, rng):
    seen[key] += 1
    if len(pools[key]) < quota:
        pools[key].append(item)
    elif quota:
        index = rng.randrange(seen[key])
        if index < quota:
            pools[key][index] = item


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build(input_dir, output_dir, n_entities=10000, extra_records=10000, seed=42):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"Output directory already exists; refusing to overwrite: {output_dir}")
    if n_entities <= 0 or extra_records < 0:
        raise ValueError("n-entities must be positive and extra-records nonnegative")
    inputs = {f"source{i}": input_dir / f"train_source{i}.tsv" for i in (1, 2, 3)}
    inputs["ground_truth"] = input_dir / "train_ground_truth.tsv"
    metadata = {key: {"path": str(path), "size_bytes": path.stat().st_size,
                      "mtime_ns": path.stat().st_mtime_ns} for key, path in inputs.items()}
    output_dir.mkdir(parents=True)
    marker = output_dir / "INCOMPLETE.txt"
    marker.write_text("Preparation is incomplete. Do not use this dataset.\n", encoding="utf-8")
    started = time.perf_counter()
    rng = random.Random(seed)
    try:
        print("Indexing Source 1 and ground truth on temporary disk...", flush=True)
        with tempfile.TemporaryDirectory(prefix="er_subset_") as temporary:
            with closing(sqlite3.connect(str(Path(temporary) / "index.sqlite"))) as db:
                db.execute("PRAGMA journal_mode=OFF")
                db.execute("PRAGMA synchronous=OFF")
                db.execute("CREATE TABLE entities (id TEXT PRIMARY KEY, country TEXT, bucket TEXT)")
                batch = []
                for row in rows(inputs["source1"], SOURCE_HEADER):
                    if not row[0].startswith("S1-"):
                        raise ValueError(f"Invalid Source 1 ID: {row[0]}")
                    batch.append((row[0], row[3]))
                    if len(batch) == 10000:
                        db.executemany("INSERT INTO entities(id,country) VALUES (?,?)", batch)
                        batch.clear()
                db.executemany("INSERT INTO entities(id,country) VALUES (?,?)", batch)
                batch = []

                def update_batch():
                    before = db.total_changes
                    db.executemany("UPDATE entities SET bucket=? WHERE id=? AND bucket IS NULL", batch)
                    if db.total_changes - before != len(batch):
                        raise ValueError("Ground truth has duplicate or unknown Source 1 IDs")
                    batch.clear()

                for record_id, value in rows(inputs["ground_truth"], TRUTH_HEADER):
                    count = len(targets(value))
                    batch.append((str(count) if count < 5 else "5+", record_id))
                    if len(batch) == 10000:
                        update_batch()
                update_batch()
                if db.execute("SELECT COUNT(*) FROM entities WHERE bucket IS NULL").fetchone()[0]:
                    raise ValueError("Source 1 records are missing ground-truth rows")
                counts = {(country, bucket): count for country, bucket, count in db.execute(
                    "SELECT country,bucket,COUNT(*) FROM entities GROUP BY country,bucket")}
                quotas = allocate(counts, n_entities)
                pools, seen = collections.defaultdict(list), collections.Counter()
                for record_id, country, bucket in db.execute("SELECT id,country,bucket FROM entities ORDER BY rowid"):
                    key = (country, bucket)
                    reservoir(pools, seen, key, record_id, quotas[key], rng)
        assignment, split_sizes = partition(pools, rng)
        selected = set(assignment)
        expected = {name: {} for name in inputs}
        selected_targets = {}
        for row in rows(inputs["ground_truth"], TRUTH_HEADER):
            if row[0] in selected:
                expected["ground_truth"][row[0]] = row
                for target in targets(row[1]):
                    if target in selected_targets:
                        raise ValueError(f"Shared target ID in selected ground truth: {target}")
                    selected_targets[target] = row[0]
        # Also catch selected targets belonging to businesses outside the subset.
        for record_id, value in rows(inputs["ground_truth"], TRUTH_HEADER):
            if record_id not in selected:
                for target in value.split(","):
                    if target in selected_targets:
                        raise ValueError(f"Selected target {target} is also linked to unselected business {record_id}")
        print(f"Selected {len(selected):,} businesses and {len(selected_targets):,} true matches; sampling extras...", flush=True)
        eligible = collections.Counter()
        for source in ("source2", "source3"):
            for row in rows(inputs[source], SOURCE_HEADER):
                if row[0] not in selected_targets:
                    eligible[(source, row[3])] += 1
        extra_quotas = allocate(eligible, extra_records)
        extra_pools, seen = collections.defaultdict(list), collections.Counter()
        for source in ("source2", "source3"):
            for row in rows(inputs[source], SOURCE_HEADER):
                if not row[0].startswith("S2-" if source == "source2" else "S3-"):
                    raise ValueError(f"Invalid {source} ID: {row[0]}")
                if row[0] in selected_targets:
                    if row[0] in expected[source]:
                        raise ValueError(f"Duplicate selected record: {row[0]}")
                    expected[source][row[0]] = row
                else:
                    key = (source, row[3])
                    reservoir(extra_pools, seen, key, row, extra_quotas[key], rng)
        extra_groups = {key: [row[0] for row in records] for key, records in extra_pools.items()}
        extras, extra_sizes = partition(extra_groups, rng)
        for (source, _), records in extra_pools.items():
            for row in records:
                if row[0] in expected[source]:
                    raise ValueError(f"Duplicate extra record: {row[0]}")
                expected[source][row[0]] = row
        if len(extras) != extra_records:
            raise ValueError("Extra sample has duplicate IDs or incorrect size")
        for row in rows(inputs["source1"], SOURCE_HEADER):
            if row[0] in selected:
                expected["source1"][row[0]] = row
        found_targets = set(expected["source2"]) | set(expected["source3"])
        missing = set(selected_targets) - found_targets
        if missing:
            raise ValueError(f"Missing referenced records: {sorted(missing)[:5]}")
        record_assignment = dict(assignment)
        record_assignment.update({target: assignment[owner] for target, owner in selected_targets.items()})
        record_assignment.update(extras)
        print("Writing TSV files and verifying exported fields and split isolation...", flush=True)
        output_counts, hashes = {}, {}
        all_seen = set()
        for split in SPLITS:
            folder = output_dir / split
            folder.mkdir()
            output_counts[split] = {}
            split_ids = {}
            for source, originals in expected.items():
                header = TRUTH_HEADER if source == "ground_truth" else SOURCE_HEADER
                path = folder / f"{source}.tsv"
                chosen = sorted(record_id for record_id in originals if record_assignment[record_id] == split)
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                    writer.writerow(header)
                    writer.writerows(originals[record_id] for record_id in chosen)
                restored = list(rows(path, header))
                ids = [row[0] for row in restored]
                if len(ids) != len(set(ids)) or ids != chosen:
                    raise ValueError(f"Duplicate or incorrect IDs in {path}")
                if any(row != originals[row[0]] for row in restored):
                    raise ValueError(f"Exported fields changed in {path}")
                split_ids[source] = set(ids)
                if source != "ground_truth":
                    if all_seen.intersection(ids):
                        raise ValueError("Record IDs overlap across sources or splits")
                    all_seen.update(ids)
                output_counts[split][source] = len(ids)
                hashes[f"{split}/{source}.tsv"] = digest(path)
            if split_ids["source1"] != split_ids["ground_truth"] or len(split_ids["source1"]) != split_sizes[split]:
                raise ValueError(f"Source 1 / ground truth count mismatch in {split}")
            for record_id in split_ids["ground_truth"]:
                if not set(targets(expected["ground_truth"][record_id][1])) <= (split_ids["source2"] | split_ids["source3"]):
                    raise ValueError(f"True matches missing from split: {record_id}")
            if sum(value == split for value in extras.values()) != extra_sizes[split]:
                raise ValueError(f"Incorrect extra count in {split}")
        strata = []
        for (country, bucket), population in sorted(counts.items()):
            members = pools.get((country, bucket), [])
            strata.append({"country": country, "match_count": bucket, "original": population,
                           "selected": len(members), "splits": dict(collections.Counter(assignment[item] for item in members))})
        for key, path in inputs.items():
            stat = path.stat()
            if stat.st_size != metadata[key]["size_bytes"] or stat.st_mtime_ns != metadata[key]["mtime_ns"]:
                raise ValueError(f"Input changed during preparation: {path}")
        manifest = {"status": "verified", "settings": {"n_entities": n_entities, "extra_records": extra_records, "seed": seed},
                    "inputs": metadata, "selected_businesses": dict(sorted(assignment.items())),
                    "extra_records": dict(sorted(extras.items())), "strata": strata,
                    "extra_strata": [{"source": key[0], "country": key[1], "eligible": eligible[key], "selected": extra_quotas[key]} for key in sorted(eligible)],
                    "output_counts": output_counts, "tsv_sha256": hashes,
                    "verification": {"all_checks_passed": True, "original_fields_preserved": True,
                                     "complete_true_matches": True, "disjoint_splits": True},
                    "preparation_seconds": round(time.perf_counter() - started, 3)}
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output_dir / "README.md").write_text(make_readme(manifest), encoding="utf-8")
        marker.unlink()
        print(json.dumps({"status": "verified", "output": str(output_dir), "counts": output_counts,
                          "seconds": manifest["preparation_seconds"]}, indent=2), flush=True)
        return manifest
    except Exception as error:
        marker.write_text(f"Preparation failed. Do not use this dataset.\n{type(error).__name__}: {error}\n", encoding="utf-8")
        raise


def make_readme(manifest):
    settings = manifest["settings"]
    return f"""# Reduced entity-resolution lab dataset

Contains {settings['n_entities']:,} labeled Source 1 businesses and all their true
matches, plus {settings['extra_records']:,} random Source 2/3 distractor records.
The original competition data is preserved. All files are UTF-8 TSV; use sep='\\t'.

## Usage

Use train/ for fitting, validation/ for threshold and model selection, and
evaluation/ only for final local assessment. Each contains source1.tsv,
source2.tsv, source3.tsv, and ground_truth.tsv with the competition headers.
Empty matched_entity_ids means no true matches. Source 2/3 records belonging to
other businesses and the random extras are incorrect alternatives, not additional
positive labels. Pairwise training examples have not yet been generated.

Businesses are stratified jointly by country and match count (0, 1, 2, 3, 4, 5+).
Splits have exact 70/15/15 totals (rounded for custom sizes), with approximately
proportional strata. True matches stay with their business. Extras are stratified
by source and country and partitioned independently in the same proportions.
No record ID is shared between splits.

## Regeneration

From the project root, choose a NEW output directory:

```powershell
python lab_project/scripts/create_lab_subset.py --n-entities {settings['n_entities']} --extra-records {settings['extra_records']} --seed {settings['seed']} --output-dir lab_project/data/dataset_small_10k_repeat
```

The defaults read student_resource/dataset/train and write
lab_project/data/dataset_small_10k, resolved relative to the script.
Explicit relative paths are resolved relative to your working directory.
Existing output directories are never overwritten. INCOMPLETE.txt means preparation
failed or was interrupted; do not use that directory. Temporary disk indexing keeps
the full Source 1 index out of RAM; allow temporary disk space for the index.

manifest.json records membership, original/sampled joint distributions, counts,
input metadata, output SHA-256 hashes, settings, and verification results. The same
inputs, seed, and script produce identical TSV bytes; elapsed time and input paths
in the manifest may differ. Original text field values are preserved, although TSV
quoting and line endings can differ from the original files.

## Limits

A reduced candidate pool makes matching easier. Random distractors are not guaranteed
to be hard negatives. Lab metrics are not competition performance estimates.
Later, assess blocking recall and evaluate held-out businesses against a larger
target pool. Do not use entity IDs as predictive features. Evaluate macro F_0.5 per
Source 1 business, including singletons. The competition test set was not used.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-entities", type=int, default=10000)
    parser.add_argument("--extra-records", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        build(args.input_dir, args.output_dir, args.n_entities, args.extra_records, args.seed)
    except (ValueError, OSError, sqlite3.IntegrityError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
