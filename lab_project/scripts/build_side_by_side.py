"""Export an offline comparison from frozen predictions/weights; no fitting/inference."""

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import polars as pl
from plotly.offline import get_plotlyjs
from cleaning import clean_for_retrieval

PROJECT=Path(__file__).resolve().parents[1]
RUN=PROJECT/'artifacts/model_comparison_v1'
REPORT=PROJECT/'reports/model_comparison'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    meta=json.loads((RUN/'run.json').read_text(encoding='utf-8'))
    if meta['status']!='complete':raise ValueError('Saved model experiment incomplete')
    for name,expected in meta['artifact_sha256'].items():
        if sha(RUN/name)!=expected:raise ValueError(f'Frozen artifact changed: {name}')
    original=json.loads((REPORT/'walkthrough_data.json').read_text(encoding='utf-8'))
    protocol=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
    variants=list(dict.fromkeys(p['variant'] for p in original['pipelines']))
    frames={v:pl.read_parquet(RUN/'features'/v/'validation.parquet') for v in variants}
    lookups={v:{pair:i for i,pair in enumerate(frame.select('source1_entity_id','target_id').iter_rows())} for v,frame in frames.items()}
    common=set.intersection(*(set(index) for index in lookups.values()))
    shared=sorted(common)
    models=[];reference_arrays={}
    for pipeline in original['pipelines']:
        variant=pipeline['variant'];pid=pipeline['pipeline_id'];folder=RUN/'models'/pid
        contributions=np.load(folder/'validation_contributions.npz')['values']
        indices=[lookups[variant][pair] for pair in shared]
        predictions=pl.read_parquet(folder/'validation_predictions.parquet')
        assert predictions.select('source1_entity_id','target_id').equals(frames[variant].select('source1_entity_id','target_id')), 'Prediction / feature row order mismatch'
        assert contributions.shape==(len(predictions),len(original['features'])+1)
        scores=predictions['score'].to_numpy()
        assert np.allclose(1/(1+np.exp(-contributions.sum(axis=1))),scores,atol=3e-6,rtol=0), 'Contributions do not reconstruct saved scores'
        reference_arrays[pid]=(scores,contributions)
        model={key:pipeline[key] for key in ['pipeline_id','variant','model','label','variant_label','model_label','threshold','metrics']}
        saved_metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        assert model['metrics']==saved_metrics and model['threshold']==saved_metrics['threshold']
        model['mean_absolute_contribution']=np.abs(contributions[indices,:-1]).mean(axis=0).tolist()
        if pipeline['model']=='logistic_regression':
            with (folder/'model.pkl').open('rb') as stream:saved=pickle.load(stream)
            weights=pipeline['training']
            assert np.allclose(weights['coefficients'],saved.steps[1][1].coef_[0],atol=1e-12)
            assert np.allclose(weights['mean'],saved.steps[0][1].mean_,atol=1e-12)
            assert np.allclose(weights['scale'],saved.steps[0][1].scale_,atol=1e-12)
            assert weights['intercept']==float(saved.steps[1][1].intercept_[0])
            model['training']={key:weights[key] for key in ['coefficients','mean','scale','intercept','optimizer_iterations','training_pairs','positive_pairs','negative_pairs','C']}
        else:
            model['training']={'trees':len(pipeline['training']['trees']),'max_depth':protocol['models']['xgboost']['max_depth'],
                'learning_rate':pipeline['training']['learning_rate'],'lambda':pipeline['training']['lambda'],
                'training_pairs':pipeline['training']['training_pairs']}
        models.append(model)
    cases=[]
    for case in original['cases']:
        item={key:case[key] for key in ['id','record','truth']};item['variants']={}
        for variant in variants:
            pairs=[]
            for pair in case['variants'][variant]['pairs']:
                index=lookups[variant][(case['id'],pair['target_id'])]
                row=frames[variant].row(index,named=True)
                assert pair['features']==[float(row[f]) for f in original['features']]
                assert pair['is_match']==row['is_match']
                for name in ['logistic_regression','xgboost']:
                    scores,contrib=reference_arrays[variant+'__'+name]
                    assert pair['scores'][name]==float(scores[index])
                    assert np.allclose(pair['contributions'][name],contrib[index],atol=1e-6)
                pairs.append({key:pair[key] for key in ['target_id','record','is_match','features','scores','contributions','ranks']})
            item['variants'][variant]={'pairs':pairs,'reference_view':case['variants'][variant]['reference_view']}
        cases.append(item)
    # Include actual disagreements, selected deterministically from saved validation
    # predictions. Existing walkthrough examples happen to have matching decisions.
    disagreements=[]
    first_variant=original['selection']['variant']
    for variant in [first_variant,*[v for v in variants if v!=first_variant]]:
        lr=next(m for m in models if m['variant']==variant and m['model']=='logistic_regression')
        xgb=next(m for m in models if m['variant']==variant and m['model']=='xgboost')
        a=reference_arrays[lr['pipeline_id']][0]>=lr['threshold']
        b=reference_arrays[xgb['pipeline_id']][0]>=xgb['threshold']
        differing=frames[variant].filter(pl.Series(a!=b)).sort('source1_entity_id','target_id')
        if len(differing):
            row=differing.row(0,named=True)
            disagreements.append((variant,row['source1_entity_id'],row['target_id']))
    dataset=PROJECT/'data/dataset_small_10k/validation'
    records={}
    for source in ['source1','source2','source3']:
        records.update({row['entity_id']:row for row in pl.read_csv(dataset/f'{source}.tsv',separator='\t',infer_schema=False).to_dicts()})
    truths={key:value.split(',') if value else [] for key,value in pl.read_csv(dataset/'ground_truth.tsv',separator='\t',infer_schema=False).iter_rows()}
    for _,key,_ in disagreements:
        if any(c['id']==key for c in cases):continue
        item={'id':key,'record':records[key],'truth':truths[key],'purpose':'Model disagreement example','variants':{}}
        for variant in variants:
            pairs=[]
            for row in frames[variant].filter(pl.col('source1_entity_id')==key).to_dicts():
                target=row['target_id'];i=lookups[variant][(key,target)]
                pairs.append({'target_id':target,'record':records[target],'is_match':row['is_match'],
                    'features':[float(row[f]) for f in original['features']], 'ranks':{},
                    'scores':{name:float(reference_arrays[variant+'__'+name][0][i]) for name in ['logistic_regression','xgboost']},
                    'contributions':{name:reference_arrays[variant+'__'+name][1][i].tolist() for name in ['logistic_regression','xgboost']}})
            item['variants'][variant]={'pairs':pairs,'reference_view':{
                field:clean_for_retrieval(records[key][column],column,variant)
                for field,column in [('name','business_name'),('address','business_address')]}}
        cases.append(item)
    payload={'features':original['features'],'models':models,'variants':variants,'cases':cases,
        'default_variant':original['selection']['variant'],
        'default_case':disagreements[0][1] if disagreements else cases[0]['id'],
        'default_target':disagreements[0][2] if disagreements else None,
        'global_cohort':{'pairs':len(shared),'scope':'validation pairs retrieved by all three preprocessing variants',
            'measure':'mean absolute per-feature contribution to raw log-odds',
            'sha256':hashlib.sha256(json.dumps(shared,separators=(',',':')).encode()).hexdigest()},
        'provenance':{'models_retrained':False,'additional_evaluation_inference_runs':0,'view_scope':'saved validation examples only'}}
    encoded=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    (REPORT/'side_by_side_data.json').write_text(encoded,encoding='utf-8')
    template=(PROJECT/'scripts/side_by_side.html').read_text(encoding='utf-8')
    html=template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',encoded)
    target=REPORT/'side_by_side_comparison.html';target.write_text(html,encoding='utf-8')
    # Preserve discovery links when either original report is rebuilt separately.
    for name in ['interactive_walkthrough.html','model_comparison.html']:
        page=REPORT/name
        if page.exists():
            content=page.read_text(encoding='utf-8')
            if 'href="side_by_side_comparison.html"' not in content:
                start=content.index('>',content.index('<body'))+1
                nav='<nav style="padding:14px 24px;font:15px system-ui;background:#edf3fb"><a href="side_by_side_comparison.html">Compare learned weights and same-pair explanations side by side</a></nav>'
                page.write_text(content[:start]+nav+content[start:],encoding='utf-8')
    (REPORT/'side_by_side_manifest.json').write_text(json.dumps({'scope':payload['provenance'],
        'common_validation_pairs':len(shared),'cases':len(cases),'models':len(models),
        'html_sha256':hashlib.sha256(html.encode()).hexdigest(),'frozen_selection_sha256':sha(RUN/'selection.json')},indent=2)+'\n',encoding='utf-8')
    print(f'Side-by-side report: {target}\nShared contribution cohort: {len(shared):,} validation candidate pairs.',flush=True)
    return target


if __name__=='__main__':build()
