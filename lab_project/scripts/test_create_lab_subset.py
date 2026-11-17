"""Small, independent integrity scenarios for subset preparation."""

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from create_lab_subset import SOURCE_HEADER, TRUTH_HEADER, allocate, build, rows


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.sources = {name: [] for name in ("source1", "source2", "source3")}
        self.truth = []
        for i in range(20):
            country = "India" if i % 2 else "US"
            self.sources["source1"].append([f"S1-{i}", f"व्यापार café {i}", "1 Road\tSuite, étage", country])
            matches = []
            if i % 4:
                for source in (2, 3):
                    record_id = f"S{source}-{i}"
                    matches.append(record_id)
                    self.sources[f"source{source}"].append([record_id, f"Shop {i}", "Some address", country])
            self.truth.append([f"S1-{i}", ",".join(matches)])
        for source in (2, 3):
            for i in range(20):
                self.sources[f"source{source}"].append([f"S{source}-extra{i}", "Distractor", "Elsewhere", "US" if i % 2 else "India"])
        self.save()

    def save(self):
        tables = {**self.sources, "ground_truth": self.truth}
        for name, records in tables.items():
            with (self.input / f"train_{name}.tsv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(TRUTH_HEADER if name == "ground_truth" else SOURCE_HEADER)
                writer.writerows(records)

    def run_build(self, folder="output", n=20, extras=10):
        with contextlib.redirect_stdout(io.StringIO()):
            return build(self.input, self.root / folder, n, extras, 42)

    def test_complete_labels_unicode_split_counts_and_reproducibility(self):
        first = self.run_build()
        second = self.run_build("repeat")
        self.assertEqual(first["tsv_sha256"], second["tsv_sha256"])
        self.assertEqual([first["output_counts"][s]["source1"] for s in ("train", "validation", "evaluation")], [14, 3, 3])
        self.assertEqual(len(first["extra_records"]), 10)
        observed = {}
        for split in ("train", "validation", "evaluation"):
            folder = self.root / "output" / split
            s1 = {row[0]: row for row in rows(folder / "source1.tsv", SOURCE_HEADER)}
            observed.update(s1)
            truth = dict(rows(folder / "ground_truth.tsv", TRUTH_HEADER))
            target_ids = {row[0] for source in (2, 3) for row in rows(folder / f"source{source}.tsv", SOURCE_HEADER)}
            for record_id, value in truth.items():
                self.assertEqual(value, dict(self.truth)[record_id])
                self.assertTrue(set(value.split(",") if value else []) <= target_ids)
        self.assertEqual(observed, {row[0]: row for row in self.sources["source1"]})
        self.assertEqual(sum(not value for _, value in self.truth), 5)
        self.assertFalse((self.root / "output" / "INCOMPLETE.txt").exists())

    def test_missing_target_marks_incomplete(self):
        self.sources["source2"] = [row for row in self.sources["source2"] if row[0] != "S2-1"]
        self.save()
        with self.assertRaisesRegex(ValueError, "Missing referenced"):
            self.run_build()
        self.assertIn("Missing referenced", (self.root / "output" / "INCOMPLETE.txt").read_text())

    def test_shared_target_rejected(self):
        self.truth[2][1] += ",S2-1"
        self.save()
        with self.assertRaisesRegex(ValueError, "Shared target"):
            self.run_build()

    def test_selected_target_linked_to_unselected_business_rejected(self):
        self.truth = [[record_id, "S2-1"] for record_id, _ in self.truth]
        self.save()
        with self.assertRaisesRegex(ValueError, "also linked to unselected"):
            self.run_build(n=1)

    def test_duplicate_ground_truth_rejected(self):
        self.truth.append(self.truth[0])
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate or unknown"):
            self.run_build()

    def test_existing_output_is_preserved(self):
        self.run_build()
        before = (self.root / "output" / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.run_build()
        self.assertEqual(before, (self.root / "output" / "manifest.json").read_bytes())

    def test_no_extras_and_proportional_allocation(self):
        result = self.run_build(extras=0)
        self.assertEqual(result["extra_records"], {})
        self.assertEqual(allocate({"a": 6, "b": 4}, 5), {"a": 3, "b": 2})

    def test_too_few_entities_marks_incomplete(self):
        with self.assertRaisesRegex(ValueError, "Requested"):
            self.run_build(n=21)
        self.assertTrue((self.root / "output" / "INCOMPLETE.txt").exists())


if __name__ == "__main__":
    unittest.main()
