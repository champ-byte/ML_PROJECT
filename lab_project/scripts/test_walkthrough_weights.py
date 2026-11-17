"""Check displayed corpus weights and learned weight reconstructions numerically."""

from collections import Counter
import unittest

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from run_models import new_model
from walkthrough_weights import model_weights,tfidf_details,tree_path,sigmoid


class WeightExplanationTests(unittest.TestCase):
    def test_tfidf_term_counts_document_frequency_and_cosine(self):
        documents=['alpha market','alpha shop','beta market']
        vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(3,5),lowercase=False).fit(documents)
        result=tfidf_details(vectorizer,'alpha market','alpha shop',len(documents),limit=3)
        actual=vectorizer.transform(['alpha market','alpha shop'])
        self.assertAlmostEqual(result['cosine'],float(actual[0].multiply(actual[1]).sum()))
        document_counts=Counter(fragment for text in documents for fragment in set(vectorizer.build_analyzer()(text)))
        for row in result['terms']:
            self.assertEqual(row['training_df'],document_counts[row['fragment']])
            self.assertAlmostEqual(row['idf'],1+np.log(4/(1+row['training_df'])))
            self.assertAlmostEqual(row['query_weight'],row['query_raw']/result['query_norm'])
        self.assertAlmostEqual(result['shown_product_sum']+result['other_product_sum'],result['cosine'])

    def test_blank_and_unseen_fragments_do_not_create_similarity(self):
        vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(3,5),lowercase=False).fit(['alpha market'])
        result=tfidf_details(vectorizer,'漢字会社','',1)
        self.assertEqual(result['cosine'],0)
        self.assertGreater(result['query_oov_fragments'],0)

    def test_float32_split_equality_goes_to_no_branch(self):
        tree={'nodeid':0,'split':'f0','split_condition':.333333343,'yes':1,'no':2,'missing':2,
              'children':[{'nodeid':1,'leaf':-.2},{'nodeid':2,'leaf':.3}]}
        path,leaf=tree_path(tree,[float(np.float32(1/3))])
        self.assertEqual(leaf['nodeid'],2)
        self.assertEqual(path[0]['branch'],'>=')

    def test_final_coefficients_and_newton_leaf_statistics(self):
        X=np.array([[i/40]*20 for i in range(48)],dtype=np.float32)
        y=np.array([int(i>22) for i in range(48)])
        for name in ['logistic_regression','xgboost']:
            model=new_model(name);model.fit(X,y)
            weights=model_weights(model,name,X,y)
            if name=='logistic_regression':
                z=(X-np.array(weights['mean']))/np.array(weights['scale'])
                reconstructed=sigmoid(weights['intercept']+z@np.array(weights['coefficients']))
            else:
                reconstructed=np.array([sigmoid(weights['base_margin']+sum(tree_path(tree,row)[1]['leaf'] for tree in weights['trees'])) for row in X])
                for index in weights['checkpoints']:
                    for row in X:
                        _,leaf=tree_path(weights['trees'][index],row)
                        stats=weights['leaf_statistics'][str(index)][str(leaf['nodeid'])]
                        self.assertAlmostEqual(stats['expected_stored_output'],leaf['leaf'],places=5)
            np.testing.assert_allclose(reconstructed,model.predict_proba(X)[:,1],atol=3e-6)


if __name__=='__main__':unittest.main()
