"""Six frozen model comparisons, validation threshold selection, then one evaluation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import platform
import time
import warnings

import numpy as np
import polars as pl
import rapidfuzz
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import xgboost as xgb

from blocking import TfidfBlocker,candidate_map
from cleaning import VARIANTS,normalizer
from er_metrics import blocking_metrics,f05
from model_metrics import bootstrap_macro,grid_curve,predictions_at,score_summary,threshold_curve
from pair_features import FEATURES,build_features,feature_contract
from run_blocking import DATA,PROJECT,load_split,write_candidates
from run_cleaning import read_candidates

OUTPUT=PROJECT/'artifacts/model_comparison_v1'
CLEANING=PROJECT/'artifacts/cleaning_v1'
MODEL_NAMES=('logistic_regression','xgboost')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def new_model(name):
    if name=='logistic_regression':
        return make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,solver='lbfgs',random_state=42))
    if name=='xgboost':
        return xgb.XGBClassifier(n_estimators=200,max_depth=3,learning_rate=.05,min_child_weight=5,
            reg_lambda=1.,subsample=1.,colsample_bytree=1.,objective='binary:logistic',
            eval_metric='logloss',tree_method='hist',n_jobs=2,random_state=42)
    raise ValueError(name)


def explain(model,name,matrix):
    if name=='logistic_regression':
        transformed=model.steps[0][1].transform(matrix)
        estimator=model.steps[1][1]
        contributions=np.column_stack([transformed*estimator.coef_[0],np.full(len(matrix),estimator.intercept_[0])])
        margin=model.decision_function(matrix)
    else:
        data=xgb.DMatrix(matrix)
        contributions=model.get_booster().predict(data,pred_contribs=True)
        margin=model.get_booster().predict(data,output_margin=True)
    if not np.allclose(contributions.sum(axis=1),margin,rtol=3e-4,atol=3e-4):
        raise ValueError('Explanation contributions do not reconstruct model margin')
    return contributions


def write_matches(path,predictions):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream,delimiter='\t',lineterminator='\n')
        writer.writerow(['source1_entity_id','matched_entity_ids'])
        writer.writerows((key,','.join(sorted(value))) for key,value in sorted(predictions.items()))


def error_rows(truth,candidates,predictions):
    rows=[]
    for key in sorted(truth):
        for target in sorted(truth[key]-predictions[key]):
            rows.append({'source1_entity_id':key,'target_id':target,
                         'error':'missed_by_classifier' if target in candidates[key] else 'missed_by_blocking'})
        for target in sorted(predictions[key]-truth[key]):
            rows.append({'source1_entity_id':key,'target_id':target,'error':'false_merge'})
    return pl.DataFrame(rows,schema={'source1_entity_id':pl.String,'target_id':pl.String,'error':pl.String})


def slices(truth,predictions,sources):
    result=[]
    groups={}
    for row in sources['source1'].to_dicts():
        key=row['entity_id']
        count=len(truth[key])
        for label in ['country: '+str(row['country']),'true match count: '+(str(count) if count<5 else '5+')]:
            groups.setdefault(label,[]).append(key)
    for label,ids in sorted(groups.items()):
        result.append({'slice':label,'reference_businesses':len(ids),
                       **score_summary({key:truth[key] for key in ids},{key:predictions[key] for key in ids})})
    for source,prefix in [('source2','S2-'),('source3','S3-')]:
        subtruth={key:{item for item in value if item.startswith(prefix)} for key,value in truth.items()}
        subpred={key:{item for item in value if item.startswith(prefix)} for key,value in predictions.items()}
        values=score_summary(subtruth,subpred)
        # Link-only source slice; do not claim source-restricted singleton/macro scores are the competition metric.
        result.append({'slice':'target source links: '+source,'reference_businesses':len(truth),
                       **{name:values[name] for name in ('tp','fp','fn','micro_precision','micro_recall','micro_f1')}})
    return pl.DataFrame(result)


def summarize_features(frame):
    rows=[]
    for positive in (False,True):
        part=frame.filter(pl.col('is_match')==positive)
        for feature in FEATURES:
            values=part[feature].to_numpy()
            counts,edges=np.histogram(values,bins=np.linspace(0,1,21))
            for i,count in enumerate(counts):
                rows.append({'feature':feature,'is_match':positive,'bin_left':float(edges[i]),
                             'bin_right':float(edges[i+1]),'pairs':int(count),'class_pairs':len(values),
                             'fraction_within_class':float(count/len(values)) if len(values) else 0.})
    return pl.DataFrame(rows)


def evaluate_once(output,selection,model,manifest):
    # The frozen selection exists and is hash-checked before the first evaluation file read.
    if json.loads((output/'selection.json').read_text(encoding='utf-8'))!=selection:
        raise ValueError('Frozen selection changed')
    folder=output/'evaluation';folder.mkdir()
    dump(folder/'evaluation_started.json',{'selection_sha256':sha(output/'selection.json'),
                                         'purpose':'single frozen pipeline evaluation; no retuning'})
    for relative,expected in selection['frozen_artifacts'].items():
        if sha(output/relative)!=expected:raise ValueError('Frozen model or feature contract changed')
    if sha(CLEANING/'validation'/selection['variant']/'tfidf_vectorizers.pkl')!=selection['vectorizers_sha256']:
        raise ValueError('Frozen vectorizers changed')
    if sha(CLEANING/'rules.json')!=selection['cleaning_rules_sha256']:
        raise ValueError('Frozen cleaning rules changed')
    sources={};hashes={}
    for name in ('source1','source2','source3'):
        path=DATA/'evaluation'/f'{name}.tsv'
        hashes[f'evaluation/{name}.tsv']=sha(path)
        if hashes[f'evaluation/{name}.tsv']!=manifest['tsv_sha256'][f'evaluation/{name}.tsv']:
            raise ValueError('Evaluation source hash changed')
        sources[name]=pl.read_csv(path,separator='\t',infer_schema=False)
    variant=selection['variant']
    with (CLEANING/'validation'/variant/'tfidf_vectorizers.pkl').open('rb') as stream:vectorizers=pickle.load(stream)
    blocker=TfidfBlocker(normalizer=normalizer(variant));blocker.vectorizers=vectorizers
    started=time.perf_counter();prepared=blocker.prepare(sources);evidence,diagnostics=blocker.retrieve(prepared,10)
    retrieval_seconds=time.perf_counter()-started
    reference_ids=sources['source1']['entity_id'].to_list()
    candidates=candidate_map(reference_ids,evidence)
    valid_targets=set(sources['source2']['entity_id'])|set(sources['source3']['entity_id'])
    write_candidates(folder,candidates,valid_targets)
    evidence.write_parquet(folder/'retrieval_evidence.parquet');diagnostics.write_csv(folder/'query_diagnostics.csv')
    started=time.perf_counter();features=build_features(sources,candidates,vectorizers,variant)
    feature_seconds=time.perf_counter()-started
    started=time.perf_counter();scores=model.predict_proba(features.select(FEATURES).to_numpy())[:,1]
    inference_seconds=time.perf_counter()-started
    empty_truth={key:set() for key in reference_ids}
    predictions=predictions_at(features,scores,empty_truth,selection['threshold'])
    features.select('source1_entity_id','target_id').with_columns(pl.Series('score',scores)).write_parquet(folder/'predictions.parquet')
    write_matches(folder/'matching_results.tsv',predictions)
    # Ground truth is opened only after predictions have been saved.
    path=DATA/'evaluation/ground_truth.tsv';hashes['evaluation/ground_truth.tsv']=sha(path)
    if hashes['evaluation/ground_truth.tsv']!=manifest['tsv_sha256']['evaluation/ground_truth.tsv']:
        raise ValueError('Evaluation label hash changed')
    truth={key:set(value.split(',')) if value else set()
           for key,value in pl.read_csv(path,separator='\t',infer_schema=False).iter_rows()}
    if set(truth)!=set(reference_ids):raise ValueError('Incomplete evaluation labels')
    result={**score_summary(truth,predictions),'macro_f05_bootstrap_95':bootstrap_macro(truth,predictions),
            'pipeline_id':selection['pipeline_id'],'threshold':selection['threshold'],
            'blocking':blocking_metrics(truth,candidates,len(valid_targets)),
            'retrieval_seconds':retrieval_seconds,'feature_seconds':feature_seconds,'inference_seconds':inference_seconds,
            'input_hashes':hashes,'labels_opened_after_predictions_saved':True,
            'note':'Evaluation is now consumed for this frozen pipeline. It is not fresh holdout data for future retuning.'}
    dump(folder/'metrics.json',result)
    error_rows(truth,candidates,predictions).write_csv(folder/'errors.csv')
    slices(truth,predictions,sources).write_csv(folder/'slices.csv')
    return result


def run(output_dir=OUTPUT):
    output=Path(output_dir).resolve()
    if output.exists():raise ValueError(f'Output exists; refusing to rerun evaluation: {output}')
    output.mkdir(parents=True);marker=output/'INCOMPLETE.txt';marker.write_text('Model experiment incomplete.\n',encoding='utf-8')
    started=time.perf_counter()
    try:
        clean_meta=json.loads((CLEANING/'run.json').read_text(encoding='utf-8'))
        for relative,expected in clean_meta['artifact_sha256'].items():
            if sha(CLEANING/relative)!=expected:raise ValueError(f'Cleaning artifact changed: {relative}')
        manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
        training,training_truth,train_hashes=load_split(DATA,'train',manifest)
        validation,truth,val_hashes=load_split(DATA,'validation',manifest)
        if {**train_hashes,**val_hashes}!=clean_meta['input_hashes']:raise ValueError('Train/validation inputs changed')
        protocol={'seed':42,'features':feature_contract(),'blocker':'frozen name/address union K=10 per view and source',
            'variants':list(VARIANTS),'models':{'logistic_regression':{'C':1.,'max_iter':1000,'solver':'lbfgs','scaler':'training-only StandardScaler'},
                'xgboost':{'n_estimators':200,'max_depth':3,'learning_rate':.05,'min_child_weight':5,'reg_lambda':1.,
                           'subsample':1.,'colsample_bytree':1.,'tree_method':'hist','n_jobs':2}},
            'negative_sampling':False,'class_weighting':False,'early_stopping':False,
            'threshold_search':'all distinct validation scores, tied scores processed together; accept score >= threshold',
            'threshold_ties':'macro F0.5 rounded to 12 decimals, then precision, then higher threshold',
            'pipeline_selection':'validation macro F0.5, then precision, then logistic simplicity, then baseline/placeholders/expanded order',
            'no_refit_after_selection':True,'evaluation_policy':'only the frozen winning pipeline; labels read after predictions saved'}
        dump(output/'protocol.json',protocol);dump(output/'feature_contract.json',feature_contract())
        results=[];fitted={}
        with threadpool_limits(limits=2):
            for variant in VARIANTS:
                print(f'{variant}: building fixed pair features...',flush=True)
                with (CLEANING/'validation'/variant/'tfidf_vectorizers.pkl').open('rb') as stream:vectorizers=pickle.load(stream)
                train_candidates=read_candidates(CLEANING/'train'/variant/'candidate_pairs.tsv')
                val_candidates=read_candidates(CLEANING/'validation'/variant/'candidate_pairs.tsv')
                feature_start=time.perf_counter()
                train=build_features(training,train_candidates,vectorizers,variant,training_truth)
                val=build_features(validation,val_candidates,vectorizers,variant,truth)
                feature_seconds=time.perf_counter()-feature_start
                folder=output/'features'/variant;folder.mkdir(parents=True)
                train.write_parquet(folder/'train.parquet');val.write_parquet(folder/'validation.parquet')
                summarize_features(train).write_csv(folder/'training_histograms.csv')
                X=train.select(FEATURES).to_numpy();y=train['is_match'].to_numpy().astype(int);V=val.select(FEATURES).to_numpy()
                for name in MODEL_NAMES:
                    pipeline_id=variant+'__'+name;model=new_model(name)
                    model_folder=output/'models'/pipeline_id;model_folder.mkdir(parents=True)
                    fit_start=time.perf_counter()
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter('always');model.fit(X,y)
                    fit_seconds=time.perf_counter()-fit_start
                    predict_start=time.perf_counter();scores=model.predict_proba(V)[:,1]
                    inference_seconds=time.perf_counter()-predict_start
                    curve,best=threshold_curve(val,scores,truth)
                    predictions=predictions_at(val,scores,truth,best['threshold'])
                    summary=score_summary(truth,predictions)
                    if abs(summary['macro_f05']-best['macro_f05'])>1e-10:raise ValueError('Threshold sweep disagrees with set scoring')
                    row={'pipeline_id':pipeline_id,'variant':variant,'model':name,'threshold':best['threshold'],
                         **summary,'candidate_average_precision':float(average_precision_score(val['is_match'],scores)),
                         'blocking_recall':sum(len(truth[key]&val_candidates[key]) for key in truth)/sum(map(len,truth.values())),
                         'training_pairs':train.height,'validation_pairs':val.height,'feature_seconds_train_plus_validation':feature_seconds,
                         'fit_seconds':fit_seconds,'inference_seconds':inference_seconds,
                         'fit_warnings':' | '.join(str(item.message) for item in captured)}
                    results.append(row);fitted[pipeline_id]=model
                    with (model_folder/'model.pkl').open('wb') as stream:pickle.dump(model,stream)
                    if name=='xgboost':model.save_model(model_folder/'model.ubj')
                    curve.write_parquet(model_folder/'threshold_curve.parquet')
                    dump(model_folder/'display_curve.json',grid_curve(curve,best['threshold']))
                    val.select('source1_entity_id','target_id','is_match').with_columns(pl.Series('score',scores)).write_parquet(model_folder/'validation_predictions.parquet')
                    contributions=explain(model,name,V)
                    np.savez_compressed(model_folder/'validation_contributions.npz',values=contributions)
                    write_matches(model_folder/'matching_results.tsv',predictions)
                    error_rows(truth,val_candidates,predictions).write_csv(model_folder/'validation_errors.csv')
                    slices(truth,predictions,validation).write_csv(model_folder/'validation_slices.csv')
                    dump(model_folder/'metrics.json',row)
                    print(f"{pipeline_id}: validation F0.5={row['macro_f05']:.5f}, P={row['micro_precision']:.5f}, R={row['micro_recall']:.5f}, threshold={row['threshold']:.6f}",flush=True)
            pl.DataFrame(results).write_csv(output/'validation_comparison.csv')
            chosen=min(results,key=lambda row:(-round(row['macro_f05'],12),-row['micro_precision'],
                       MODEL_NAMES.index(row['model']),VARIANTS.index(row['variant'])))
            selection={'pipeline_id':chosen['pipeline_id'],'variant':chosen['variant'],'model':chosen['model'],
                       'threshold':chosen['threshold'],'selection_scope':'validation only',
                       'validation_macro_f05':chosen['macro_f05'],'feature_order':FEATURES,
                       'frozen_artifacts':{relative:sha(output/relative) for relative in
                          [f"models/{chosen['pipeline_id']}/model.pkl",'protocol.json','feature_contract.json']},
                       'vectorizers_sha256':sha(CLEANING/'validation'/chosen['variant']/'tfidf_vectorizers.pkl'),
                       'cleaning_rules_sha256':sha(CLEANING/'rules.json')}
            dump(output/'selection.json',selection)
            print(f"Frozen {chosen['pipeline_id']}. Starting its single evaluation pass...",flush=True)
            evaluation=evaluate_once(output,selection,fitted[chosen['pipeline_id']],manifest)
        result={'status':'complete','classifier_trained':True,'evaluation_consumed':True,
            'evaluation_runs':1,'selection_sha256':sha(output/'selection.json'),'selected_pipeline':chosen['pipeline_id'],
            'versions':{'python':platform.python_version(),'polars':pl.__version__,'numpy':np.__version__,
                        'sklearn':sklearn.__version__,'xgboost':xgb.__version__,'rapidfuzz':rapidfuzz.__version__},
            'input_hashes':{**train_hashes,**val_hashes,**evaluation['input_hashes']},
            'elapsed_seconds':time.perf_counter()-started,
            'script_hashes':{name:sha(PROJECT/'scripts'/name) for name in
                            ['pair_features.py','model_metrics.py','run_models.py','blocking.py','cleaning.py']},
            'artifact_sha256':{path.relative_to(output).as_posix():sha(path) for path in sorted(output.rglob('*'))
                               if path.is_file() and path.name!='INCOMPLETE.txt'}}
        dump(output/'run.json',result);marker.unlink()
        print(f"Complete: evaluation macro F0.5={evaluation['macro_f05']:.5f}; {result['elapsed_seconds']:.1f}s total.",flush=True)
        return result
    except Exception as error:
        marker.write_text(f'Experiment failed; do not retune after evaluation access: {error}\n',encoding='utf-8');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    run(parser.parse_args().output_dir)
