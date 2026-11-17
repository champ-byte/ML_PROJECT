"""Scoring/feature invariants and leakage-boundary tests before the real experiment."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from blocking import TfidfBlocker
from er_metrics import set_metrics
from model_metrics import bootstrap_macro,grid_curve,predictions_at,threshold_curve
from pair_features import FEATURES,build_features,text_features
from run_models import evaluate_once,explain,new_model
from test_blocking import frame


class FeatureTests(unittest.TestCase):
    def test_missing_fields_are_not_exact_matches(self):
        values=text_features('','')
        self.assertEqual(values[:6],[0.,0.,0.,0.,0.,0.])
        self.assertEqual(values[6:8],[1.,1.])

    def test_hand_computed_tokens_numbers_and_edit_distance(self):
        values=text_features('12 main road','12 main street')
        self.assertEqual(values[0],.5)
        self.assertEqual(values[4],1.)
        self.assertEqual(values[5],0.)
        self.assertAlmostEqual(text_features('abc','ab')[1],2/3)
        self.assertEqual(text_features('12 34','12')[4],.5)
        self.assertEqual(text_features('12 34','12')[5],1.)

    def test_feature_matrix_order_and_cosine(self):
        sources={'source1':frame([('S1-a','Alpha Market','12 Road','US')]),
                 'source2':frame([('S2-a','Alpha Market','12 Road','US')]),'source3':frame([])}
        blocker=TfidfBlocker().fit(sources)
        result=build_features(sources,{'S1-a':{'S2-a'}},blocker.vectorizers,'baseline',{'S1-a':{'S2-a'}})
        self.assertEqual(result.select(FEATURES).shape,(1,20))
        self.assertAlmostEqual(result['name_cosine'][0],1.,places=5)
        self.assertEqual(result['name_exact'][0],1.)
        self.assertEqual(result['is_match'][0],True)
        self.assertNotIn('is_match',FEATURES)
        self.assertNotIn('target_id',FEATURES)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.truth={'A':{'x','blocked'},'B':set(),'C':{'z'}}
        self.frame=pl.DataFrame({'source1_entity_id':['A','A','B','C'],'target_id':['x','wrong','other','z']})
        self.scores=np.array([.8,.8,.3,.7])

    def test_sweep_matches_brute_force_including_ties_singletons_and_blocked_misses(self):
        curve,best=threshold_curve(self.frame,self.scores,self.truth)
        for row in curve.to_dicts():
            expected=set_metrics(self.truth,predictions_at(self.frame,self.scores,self.truth,row['threshold']))
            for key in ('macro_f05','micro_precision','micro_recall','exact_set_accuracy','singleton_correctness'):
                self.assertAlmostEqual(row[key],expected[key])
        self.assertGreaterEqual(best['fn'],1)  # unretrieved true pair must remain FN
        self.assertEqual(curve.height,4)  # three unique scores plus reject-all

    def test_display_threshold_metrics_are_exact(self):
        curve,best=threshold_curve(self.frame,self.scores,self.truth)
        for row in grid_curve(curve,best['threshold']):
            expected=set_metrics(self.truth,predictions_at(self.frame,self.scores,self.truth,row['threshold']))
            self.assertAlmostEqual(row['macro_f05'],expected['macro_f05'])
            self.assertEqual(row['fp'],expected['fp'])

    def test_bootstrap_constant_scores(self):
        ci=bootstrap_macro({'A':set(),'B':{'x'}},{'A':set(),'B':{'x'}},repeats=20)
        self.assertEqual((ci['low'],ci['high']),(1.,1.))

    def test_evaluation_requires_identical_frozen_selection_before_reading_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'selection.json').write_text(json.dumps({'frozen':True}))
            with self.assertRaisesRegex(ValueError,'Frozen selection changed'):
                evaluate_once(path,{'frozen':False},None,{})
            self.assertFalse((path/'evaluation').exists())


class ExplanationTests(unittest.TestCase):
    def test_local_contributions_reconstruct_both_models(self):
        X=np.array([[i/30,(i%4)/4] for i in range(32)],dtype=np.float32)
        y=np.array([int(i>15) for i in range(32)])
        for name in ('logistic_regression','xgboost'):
            model=new_model(name);model.fit(X,y)
            values=explain(model,name,X)
            self.assertEqual(values.shape,(32,3))
            self.assertTrue(np.isfinite(values).all())


if __name__=='__main__':unittest.main()
