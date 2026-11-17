"""Fixed, bounded pair features. IDs/labels never enter the feature matrix."""

import re

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

from cleaning import clean_for_retrieval


FEATURES = [f'{field}_{metric}' for field in ('name','address') for metric in
            ('cosine','token_jaccard','edit_similarity','length_ratio','exact',
             'number_jaccard','number_conflict','missing_left','missing_right','token_containment')]


def text_features(left, right):
    a,b=set(left.split()),set(right.split())
    na,nb=set(re.findall(r'\d+',left)),set(re.findall(r'\d+',right))
    both=bool(left and right)
    return [len(a&b)/len(a|b) if a and b else 0.,
            Levenshtein.normalized_similarity(left,right) if both else 0.,
            min(len(left),len(right))/max(len(left),len(right)) if both else 0.,
            float(both and left==right),
            len(na&nb)/len(na|nb) if na and nb else 0.,
            float(bool(na and nb) and na!=nb),float(not left),float(not right),
            len(a&b)/min(len(a),len(b)) if a and b else 0.]


def build_features(sources,candidates,vectorizers,variant,truth=None,batch_size=2048):
    reference=sources['source1'].sort('entity_id')
    target=pl.concat([sources['source2'],sources['source3']]).sort('entity_id')
    query_ids=reference['entity_id'].to_list();target_ids=target['entity_id'].to_list()
    query_lookup={key:i for i,key in enumerate(query_ids)};target_lookup={key:i for i,key in enumerate(target_ids)}
    if set(candidates)!=set(query_ids):raise ValueError('Candidates must include every reference')
    pairs=[(key,item) for key in sorted(candidates) for item in sorted(candidates[key])]
    if any(item not in target_lookup for _,item in pairs):raise ValueError('Unknown candidate target')
    qi=np.array([query_lookup[key] for key,_ in pairs],dtype=np.int64)
    ti=np.array([target_lookup[item] for _,item in pairs],dtype=np.int64)
    matrix=np.zeros((len(pairs),len(FEATURES)),dtype=np.float32)
    for field_index,(view,field) in enumerate([('name','business_name'),('address','business_address')]):
        left=[clean_for_retrieval(value,field,variant) for value in reference[field]]
        right=[clean_for_retrieval(value,field,variant) for value in target[field]]
        q=vectorizers[view].transform(left).tocsr();t=vectorizers[view].transform(right).tocsr()
        base=field_index*10
        for offset in range(0,len(pairs),batch_size):
            end=min(offset+batch_size,len(pairs))
            matrix[offset:end,base]=np.asarray(q[qi[offset:end]].multiply(t[ti[offset:end]]).sum(axis=1)).reshape(-1)
        matrix[:,base]=np.clip(matrix[:,base],0,1)
        for i,(a,b) in enumerate(zip(qi,ti)):
            matrix[i,base+1:base+10]=text_features(left[a],right[b])
    if not np.isfinite(matrix).all() or (matrix<0).any() or (matrix>1.000001).any():
        raise ValueError('Invalid feature values')
    frame=pl.DataFrame({'source1_entity_id':[key for key,_ in pairs], 'target_id':[item for _,item in pairs],
                        **{name:matrix[:,i] for i,name in enumerate(FEATURES)}})
    if truth is not None:
        frame=frame.with_columns(pl.Series('is_match',[item in truth[key] for key,item in pairs],dtype=pl.Boolean))
    return frame


def feature_contract():
    return {'features':FEATURES,'range':[0,1],'missing_similarity':0,
            'exact_requires_both_present':True,'edit_similarity':'normalized Levenshtein with unit edit costs',
            'number_tokens':'Unicode digit runs, leading zeros retained',
            'number_conflict':'both digit sets present and unequal',
            'token_containment':'intersection / smaller nonempty token-set size',
            'token_jaccard':'intersection / union, zero when either set is empty',
            'text_view':'same frozen cleaning variant as blocking',
            'excluded':['IDs','labels','manifest distractor membership','country one-hot','source labels']}
