"""Hand-computed scoring and retrieval invariants on small independent fixtures."""

import unittest

import numpy as np
import polars as pl

from blocking import TfidfBlocker, candidate_map, exact_candidates, normalize_text
from er_metrics import blocking_metrics, f05, set_metrics


def frame(rows):
    return pl.DataFrame(rows, schema={"entity_id": pl.String, "business_name": pl.String,
                                     "business_address": pl.String, "country": pl.String}, orient="row")


class MetricsTests(unittest.TestCase):
    def test_competition_example_and_singletons(self):
        self.assertAlmostEqual(f05({'a','b'}, {'a','b','x'}), 5/7)
        self.assertEqual(f05(set(),set()), 1)
        self.assertEqual(f05(set(),{'x'}), 0)
        self.assertEqual(f05({'x'},set()), 0)

    def test_hand_computed_end_to_end_and_blocking_oracle(self):
        truth = {'A': {'S2-a','S3-b'}, 'B': {'S2-c'}, 'C': set()}
        candidates = {'A': {'S2-a','S2-x'}, 'B': set(), 'C': {'S2-x'}}
        end = set_metrics(truth,candidates)
        self.assertEqual((end['tp'],end['fp'],end['fn']),(1,2,2))
        self.assertAlmostEqual(end['macro_f05'],1/6)
        self.assertAlmostEqual(end['micro_precision'],1/3)
        self.assertEqual(end['singleton_correctness'],0)
        block = blocking_metrics(truth,candidates,4)
        self.assertAlmostEqual(block['pair_recall'],1/3)
        self.assertAlmostEqual(block['oracle_macro_f05'],11/18)
        self.assertEqual(block['reduction_ratio'],.75)
        self.assertEqual(block['missed_true_pairs'],2)

    def test_missing_reference_rejected(self):
        with self.assertRaises(ValueError):
            set_metrics({'A':set()}, {})

    def test_no_positive_labels_is_explicit(self):
        result = blocking_metrics({'A':set()}, {'A':set()}, 0)
        self.assertIsNone(result['pair_recall'])
        self.assertIsNone(result['complete_match_coverage'])
        self.assertEqual(result['oracle_macro_f05'],1)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.sources = {
            'source1': frame([('S1-a','Alpha Market','10 Main Road','US'),
                              ('S1-b','Beta Store','5 River Road','India')]),
            'source2': frame([('S2-b','Alpha Market','10 Main Road','US'),
                              ('S2-a','Alpha Market','10 Main Road','US'),
                              ('S2-c','Beta Store','5 River Road','India'),
                              ('S2-z','Unrelated','Elsewhere','US')]),
            'source3': frame([('S3-a','Alpha Shop','10 Main Road','US'),
                              ('S3-b','Beta Shop','5 River Road','India')])}
        self.blocker = TfidfBlocker(batch_size=1).fit(self.sources)

    def test_ties_nested_k_and_repeatable_ranking(self):
        prepared=self.blocker.prepare(self.sources)
        one,_=self.blocker.retrieve(prepared,1)
        two,_=self.blocker.retrieve(prepared,2)
        again,_=self.blocker.retrieve(prepared,2)
        self.assertTrue(two.equals(again))
        first=one.filter((pl.col('source1_entity_id')=='S1-a') & (pl.col('target_source')=='source2') & (pl.col('view')=='name'))
        self.assertEqual(first['target_id'].to_list(),['S2-a'])
        a=candidate_map(['S1-a','S1-b'],one);b=candidate_map(['S1-a','S1-b'],two)
        self.assertTrue(all(a[key]<=b[key] for key in a))

    def test_sparse_scores_match_direct_cosine_and_country_exclusion(self):
        prepared=self.blocker.prepare(self.sources)
        evidence,_=self.blocker.retrieve(prepared,10)
        for row in evidence.to_dicts():
            i=prepared.sources['source1']['entity_id'].to_list().index(row['source1_entity_id'])
            j=prepared.sources[row['target_source']]['entity_id'].to_list().index(row['target_id'])
            expected=(prepared.matrices['source1'][row['view']][i] @ prepared.matrices[row['target_source']][row['view']][j].T).toarray()[0,0]
            self.assertAlmostEqual(row['cosine'],float(expected),places=5)
            self.assertEqual(prepared.sources['source1']['country'][i],prepared.sources[row['target_source']]['country'][j])
        self.assertNotIn('S2-c',candidate_map(['S1-a','S1-b'],evidence)['S1-a'])

    def test_blank_and_out_of_vocabulary_queries_return_no_arbitrary_targets(self):
        sources={**self.sources,'source1':frame([('S1-blank','','',None),('S1-oov','漢字会社','漢字会社','US')])}
        evidence,diag=self.blocker.retrieve(self.blocker.prepare(sources),3)
        self.assertEqual(evidence.height,0)
        self.assertTrue(all(diag['zero_vector'].to_list()))
        self.assertEqual(candidate_map(['S1-blank','S1-oov'],evidence),{'S1-blank':set(),'S1-oov':set()})

    def test_unseen_country_and_absent_country_fallback_without_refitting(self):
        before={view:dict(vectorizer.vocabulary_) for view,vectorizer in self.blocker.vectorizers.items()}
        sources={**self.sources,
                 'source1':frame([('S1-fr','Alpha Market','10 Main Road','France'),('S1-new','Beta Store','5 River Road','NeverSeen'),('S1-blank','Alpha Market','10 Main Road',None)]),
                 'source2':frame([('S2-fr','Alpha Market','10 Main Road','France'),('S2-us','Alpha Market','10 Main Road','US'),('S2-in','Beta Store','5 River Road','India')])}
        evidence,diag=self.blocker.retrieve(self.blocker.prepare(sources),10)
        result=candidate_map(sources['source1']['entity_id'],evidence)
        self.assertIn('S2-fr',result['S1-fr'])
        self.assertNotIn('S2-us',result['S1-fr'])
        self.assertIn('S2-in',result['S1-new'])
        self.assertIn('S2-us',result['S1-blank'])
        self.assertIn('global_fallback',diag['country_scope'].to_list())
        self.assertEqual(before,{view:dict(vectorizer.vocabulary_) for view,vectorizer in self.blocker.vectorizers.items()})

    def test_empty_target_source_and_blank_exact_names(self):
        sources={**self.sources,'source3':frame([]),'source1':frame([('S1-blank','',None,'US'),('S1-a','  ALPHA MARKET! ','','US')])}
        evidence,_=self.blocker.retrieve(self.blocker.prepare(sources),10)
        self.assertNotIn('source3',evidence['target_source'].to_list())
        exact=exact_candidates(sources)
        self.assertEqual(exact['S1-blank'],set())
        self.assertEqual(exact['S1-a'],{'S2-a','S2-b'})

    def test_unicode_normalization_preserves_scripts_and_accents(self):
        self.assertEqual(normalize_text('  CAFÉ & राम  '),'café राम')


if __name__=='__main__':
    unittest.main()
