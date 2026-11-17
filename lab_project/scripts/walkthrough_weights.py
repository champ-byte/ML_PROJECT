"""Reconstruct explanations from frozen retrieval/model objects; never fit models."""

from collections import Counter
import json

import numpy as np
from sklearn.metrics import log_loss
import xgboost as xgb

from pair_features import FEATURES


def sigmoid(value):
    return 1/(1+np.exp(-np.asarray(value)))


def tfidf_details(vectorizer,query,target,training_documents,limit=12):
    if vectorizer.sublinear_tf or not vectorizer.smooth_idf or vectorizer.norm!='l2':
        raise ValueError('This explanation expects raw counts, smoothed IDF, and L2 normalization')
    analyzer=vectorizer.build_analyzer();qcounts=Counter(analyzer(query));tcounts=Counter(analyzer(target))
    q=vectorizer.transform([query]).tocsr();t=vectorizer.transform([target]).tocsr()
    vocabulary=vectorizer.vocabulary_;idf=vectorizer.idf_
    qw=dict(zip(q.indices,q.data));tw=dict(zip(t.indices,t.data))
    known_q={word:count for word,count in qcounts.items() if word in vocabulary}
    known_t={word:count for word,count in tcounts.items() if word in vocabulary}
    qnorm=float(np.sqrt(sum((count*float(idf[vocabulary[word]]))**2 for word,count in known_q.items())))
    tnorm=float(np.sqrt(sum((count*float(idf[vocabulary[word]]))**2 for word,count in known_t.items())))
    words=sorted(set(known_q)|set(known_t),key=lambda word:(
        -float(qw.get(vocabulary[word],0)*tw.get(vocabulary[word],0)),
        -float(qw.get(vocabulary[word],0)),word))[:limit]
    rows=[]
    for word in words:
        i=vocabulary[word];weight=float(idf[i])
        # Recover integer DF from the saved smoothed IDF. No validation corpus is used.
        df=int(round((1+training_documents)/np.exp(weight-1)-1))
        a=float(qw.get(i,0));b=float(tw.get(i,0))
        rows.append({'fragment':word,'training_df':df,'idf':weight,'query_tf':qcounts.get(word,0),
            'target_tf':tcounts.get(word,0),'query_raw':qcounts.get(word,0)*weight,
            'target_raw':tcounts.get(word,0)*weight,'query_weight':a,'target_weight':b,'product':a*b})
    cosine=sum(float(qw[i])*float(tw[i]) for i in set(qw)&set(tw))
    return {'training_documents':training_documents,'query_text':query,'target_text':target,
        'query_norm':qnorm,'target_norm':tnorm,'query_generated_fragments':len(qcounts),
        'query_known_fragments':len(known_q),'query_oov_fragments':len(qcounts)-len(known_q),
        'shared_fragments':len(set(q.indices)&set(t.indices)),'cosine':cosine,
        'shown_product_sum':sum(row['product'] for row in rows),
        'other_product_sum':cosine-sum(row['product'] for row in rows),'terms':rows,
        'df_source':'integer DF recovered from the saved training-only smooth IDF'}


def tree_path(tree,features):
    node=tree;path=[]
    while 'leaf' not in node:
        name=node['split'];index=int(name[1:]) if name.startswith('f') and name[1:].isdigit() else FEATURES.index(name)
        value=float(features[index]);threshold=float(np.float32(node['split_condition']))
        child=node['missing'] if np.isnan(value) else node['yes'] if value<threshold else node['no']
        path.append({'node':node['nodeid'],'feature':FEATURES[index],'value':value,'threshold':threshold,
                     'branch':'missing' if np.isnan(value) else '<' if value<threshold else '>=',
                     'next_node':child,'gain':node.get('gain'),'cover':node.get('cover')})
        node=next(item for item in node['children'] if item['nodeid']==child)
    return path,node


def normalize_tree_thresholds(node):
    if 'split_condition' in node:
        node['split_condition']=float(np.float32(node['split_condition']))
        for child in node['children']:normalize_tree_thresholds(child)
    return node


def model_weights(model,name,train_matrix,labels):
    n=len(labels);positive=int(np.sum(labels));prior=positive/n
    common={'training_pairs':n,'positive_pairs':positive,'negative_pairs':n-positive,
            'constant_prior_log_loss':float(log_loss(labels,np.full(n,prior))),
            'feature_order':FEATURES,'fit_scope':'training candidates only'}
    if name=='logistic_regression':
        scaler,estimator=model.steps[0][1],model.steps[1][1]
        return {**common,'kind':name,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'coefficients':estimator.coef_[0].tolist(),'intercept':float(estimator.intercept_[0]),
            'optimizer_iterations':int(estimator.n_iter_[0]),'C':float(estimator.C),
            'training_log_loss':float(log_loss(labels,model.predict_proba(train_matrix)[:,1])),
            'history_note':'Final saved coefficients and iteration count; individual L-BFGS updates were not logged.'}
    booster=model.get_booster();config=json.loads(booster.save_config())
    base=config['learner']['learner_model_param']['base_score']
    base_score=float(json.loads(base)[0] if base.startswith('[') else base)
    base_margin=float(np.log(base_score/(1-base_score)))
    trees=[normalize_tree_thresholds(json.loads(value)) for value in booster.get_dump(dump_format='json',with_stats=True)]
    eta=float(model.get_params()['learning_rate']);regularization=float(model.get_params()['reg_lambda'])
    data=xgb.DMatrix(train_matrix);statistics={};loss_curve=[{'round':0,'loss':common['constant_prior_log_loss']}]
    checkpoints=sorted(set([0,min(24,len(trees)-1),min(99,len(trees)-1),len(trees)-1]))
    for index in checkpoints:
        before=np.full(n,base_score) if index==0 else booster.predict(data,iteration_range=(0,index)).astype(float)
        gradient=before-labels;hessian=before*(1-before)
        leaf_ids=booster[index:index+1].predict(data,pred_leaf=True).reshape(-1).astype(int)
        after=booster.predict(data,iteration_range=(0,index+1))
        loss_curve.append({'round':index+1,'loss':float(log_loss(labels,after))})
        by_leaf={}
        leaf_outputs={}
        pending=[trees[index]]
        while pending:
            node=pending.pop()
            if 'leaf' in node:leaf_outputs[node['nodeid']]=node['leaf']
            else:pending.extend(node['children'])
        for leaf_id in np.unique(leaf_ids):
            mask=leaf_ids==leaf_id;g=float(gradient[mask].sum());h=float(hessian[mask].sum())
            by_leaf[str(int(leaf_id))]={'rows':int(mask.sum()),'gradient_sum':g,'hessian_sum':h,
                'unshrunk_weight':-g/(h+regularization),'expected_stored_output':eta*(-g/(h+regularization))}
            if not np.isclose(by_leaf[str(int(leaf_id))]['expected_stored_output'],leaf_outputs[int(leaf_id)],rtol=1e-4,atol=1e-5):
                raise ValueError('Reconstructed Newton leaf weight disagrees with the saved tree')
        statistics[str(index)]=by_leaf
    return {**common,'kind':name,'trees':trees,'base_score':base_score,'base_margin':base_margin,
            'learning_rate':eta,'lambda':regularization,'checkpoints':checkpoints,
            'leaf_statistics':statistics,'training_loss_curve':loss_curve,
            'history_note':'Training losses/gradients reconstructed from frozen tree prefixes; no optimizer or evaluation rerun.',
            'leaf_note':'Dumped leaf outputs already include learning-rate shrinkage; do not multiply them by eta again.',
            'shap_note':'SHAP base term is a separate expected-margin reference, not the initial boosting base margin.'}
