"""Country-aware exact blocking and batched sparse character TF-IDF retrieval."""

from collections import defaultdict
from dataclasses import dataclass
import unicodedata

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


VIEWS = {"name": "business_name", "address": "business_address"}
EVIDENCE_SCHEMA = {"source1_entity_id": pl.String, "target_id": pl.String,
                   "target_source": pl.String, "view": pl.String,
                   "rank": pl.UInt32, "cosine": pl.Float32, "country_scope": pl.String}


def normalize_text(value):
    text = unicodedata.normalize("NFKC", value or "").casefold()
    text = "".join(" " if unicodedata.category(char)[0] in "PS" else char for char in text)
    return " ".join(text.split())


def normalize_field(value, field):
    return normalize_text(value)


def country_key(value):
    return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())


def eligible_country(query_country, target_country, available_countries):
    # A new label (e.g. France) is an ordinary dynamic partition if present.
    # Blank query countries or labels absent from this target source search globally.
    return (not query_country or query_country not in available_countries or
            not target_country or target_country == query_country)


def candidate_map(reference_ids, evidence, view=None):
    result = {record_id: set() for record_id in reference_ids}
    if view:
        evidence = evidence.filter(pl.col("view") == view)
    for reference_id, target in evidence.select("source1_entity_id", "target_id").iter_rows():
        result[reference_id].add(target)
    return result


def exact_candidates(sources):
    index = defaultdict(set)
    global_index = defaultdict(set)
    countries_by_source = defaultdict(set)
    for source in ("source2", "source3"):
        for row in sources[source].to_dicts():
            key = country_key(row["country"])
            if key:
                countries_by_source[source].add(key)
            name = normalize_text(row["business_name"])
            if name:
                index[(source, key, name)].add(row["entity_id"])
                global_index[(source, name)].add(row["entity_id"])
    result = {}
    for row in sources["source1"].to_dicts():
        query_country = country_key(row["country"])
        name = normalize_text(row["business_name"])
        targets = set()
        if name:
            for source in ("source2", "source3"):
                countries = countries_by_source[source]
                if not query_country or query_country not in countries:
                    targets.update(global_index.get((source, name), set()))
                else:
                    targets.update(index.get((source, query_country, name), set()))
                    targets.update(index.get((source, "", name), set()))
        result[row["entity_id"]] = targets
    return result


@dataclass
class Prepared:
    sources: dict
    normalized: dict
    matrices: dict


class TfidfBlocker:
    def __init__(self, batch_size=64, normalizer=None):
        self.batch_size = batch_size
        self.normalizer = normalizer or normalize_field
        self.vectorizers = {}

    def fit(self, training_sources):
        for view, field in VIEWS.items():
            texts = [self.normalizer(value, field) for source in training_sources.values() for value in source[field].to_list()]
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False,
                                         min_df=1, norm="l2", dtype=np.float32)
            if not any(len(value) >= 3 for value in texts):
                raise ValueError(f"No training character ngrams for {view}")
            vectorizer.fit(texts)
            self.vectorizers[view] = vectorizer
        return self

    def prepare(self, sources):
        ordered = {source: frame.sort("entity_id") for source, frame in sources.items()}
        normalized, matrices = {}, {}
        for source, frame in ordered.items():
            normalized[source] = {view: [self.normalizer(value, field) for value in frame[field].to_list()]
                                  for view, field in VIEWS.items()}
            matrices[source] = {view: (vectorizer.transform(normalized[source][view]).tocsr() if frame.height else
                                      csr_matrix((0, len(vectorizer.vocabulary_)), dtype=np.float32))
                                for view, vectorizer in self.vectorizers.items()}
        return Prepared(ordered, normalized, matrices)

    def retrieve(self, prepared, k):
        if k <= 0:
            raise ValueError("k must be positive")
        reference = prepared.sources["source1"]
        reference_ids = reference["entity_id"].to_list()
        query_countries = [country_key(value) for value in reference["country"].to_list()]
        query_groups = defaultdict(list)
        for i, key in enumerate(query_countries):
            query_groups[key].append(i)
        evidence, diagnostics = [], []
        for source in ("source2", "source3"):
            target = prepared.sources[source]
            target_ids = target["entity_id"].to_list()
            target_countries = [country_key(value) for value in target["country"].to_list()]
            available = set(target_countries) - {""}
            for country, query_indices in sorted(query_groups.items()):
                eligible = np.array([i for i, key in enumerate(target_countries)
                                     if eligible_country(country, key, available)], dtype=np.int64)
                scope = "same_country_plus_blank" if country and country in available else "global_fallback"
                for view in VIEWS:
                    query_matrix = prepared.matrices["source1"][view]
                    target_matrix = prepared.matrices[source][view]
                    selected_targets = target_matrix[eligible].T.tocsc()
                    for offset in range(0, len(query_indices), self.batch_size):
                        batch = query_indices[offset:offset + self.batch_size]
                        # This sparse product is at most batch_size by eligible targets.
                        # Never materialize a full reference-by-target dense matrix.
                        scores = (query_matrix[batch] @ selected_targets).tocsr()
                        scores.eliminate_zeros()
                        for position, query_index in enumerate(batch):
                            row = scores.getrow(position)
                            mask = row.data > 0
                            columns, values = row.indices[mask], row.data[mask]
                            if len(values) > k:
                                cutoff = np.partition(values, len(values) - k)[len(values) - k]
                                keep = values >= cutoff
                                columns, values = columns[keep], values[keep]
                            # Targets are sorted by ID: equal-score ties resolve by ID.
                            order = np.lexsort((eligible[columns], -values))[:k]
                            for rank, item in enumerate(order, 1):
                                evidence.append((reference_ids[query_index], target_ids[eligible[columns[item]]],
                                                 source, view, rank, float(values[item]), scope))
                            diagnostics.append((reference_ids[query_index], source, view, scope,
                                                not bool(prepared.normalized["source1"][view][query_index]),
                                                bool(query_matrix.indptr[query_index + 1] == query_matrix.indptr[query_index]),
                                                len(order), len(eligible)))
        frame = pl.DataFrame(evidence, schema=EVIDENCE_SCHEMA, orient="row")
        diagnostic = pl.DataFrame(diagnostics, schema={"source1_entity_id": pl.String,
            "target_source": pl.String, "view": pl.String, "country_scope": pl.String,
            "blank_query": pl.Boolean, "zero_vector": pl.Boolean,
            "retrieved": pl.UInt32, "eligible_targets": pl.UInt32}, orient="row")
        return frame, diagnostic
