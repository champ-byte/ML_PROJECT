"""Build executed feature/model notebooks and an offline interactive walkthrough."""

import json
import os
import pickle
from pathlib import Path
import sys
from textwrap import dedent

from bs4 import BeautifulSoup
from jupyter_client import KernelManager
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import numpy as np
import polars as pl
from plotly.offline import get_plotlyjs
from threadpoolctl import threadpool_limits

from cleaning import VARIANTS,clean_for_retrieval
from pair_features import FEATURES
from run_models import OUTPUT,PROJECT,DATA,MODEL_NAMES,sha
from model_metrics import predictions_at
from walkthrough_weights import model_weights,tfidf_details,tree_path,sigmoid
from blocking import eligible_country,country_key


def cell(kind,text):
    return (nbformat.v4.new_markdown_cell if kind=='md' else nbformat.v4.new_code_cell)(dedent(text).strip())


def setup(folder):
    return cell('code',f"""
    import json
    from pathlib import Path
    import sys
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    from IPython.display import display, Markdown
    PROJECT=Path({str(PROJECT)!r})
    if not (PROJECT/'scripts/run_models.py').exists():
        PROJECT=next((p if (p/'scripts/run_models.py').exists() else p/'lab_project')
            for p in [Path.cwd(),*Path.cwd().parents]
            if (p/'scripts/run_models.py').exists() or (p/'lab_project/scripts/run_models.py').exists())
    sys.path.insert(0,str(PROJECT/'scripts'))
    from eda_helpers import Report
    from pair_features import FEATURES
    from run_models import sha
    RUN=PROJECT/'artifacts/model_comparison_v1'
    run=json.loads((RUN/'run.json').read_text(encoding='utf-8'))
    assert run['status']=='complete' and not (RUN/'INCOMPLETE.txt').exists()
    for relative,expected in run['artifact_sha256'].items():assert sha(RUN/relative)==expected
    report=Report(PROJECT/'reports/{folder}')
    print('Saved artifact hashes verified. No models or evaluation are rerun by this notebook.')
    """)


def execute(name,folder,cells):
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3 (ML lab uv environment)','language':'python','name':'python3'},
        'language_info':{'name':'python','version':sys.version.split()[0]}})
    path=PROJECT/'notebooks'/f'{name}.ipynb';nbformat.write(notebook,path)
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    try:NotebookClient(notebook,km=manager,timeout=180,startup_timeout=60,resources={'metadata':{'path':str(PROJECT)}}).execute()
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    nbformat.validate(notebook);nbformat.write(notebook,path)
    exporter=HTMLExporter();exporter.exclude_input=True
    body,_=exporter.from_notebook_node(notebook);document=BeautifulSoup(body,'html.parser')
    for script in document.find_all('script',src=True):script.decompose()
    target=PROJECT/'reports'/folder/f'{folder}.html';target.write_text(str(document),encoding='utf-8')
    print(f'Executed {path.name}; exported {target}',flush=True)


def feature_notebook():
    return [cell('md',"""
    # Pair features: what evidence does the classifier receive?

    Inspect the frozen **training candidate pairs**, before reading model decisions. IDs and
    ground truth stay separate from the fixed 20 numeric features. Positives are true matches
    actually retained by blocking; unretrieved true pairs are not injected into the training set.
    This notebook reads saved features only and does not train or rerun evaluation.
    """),setup('features'),cell('code',"""
    frame=pl.read_parquet(RUN/'features/baseline/train.parquet')
    counts=frame.group_by('is_match').len(name='candidate_pairs')
    report.table('training_class_balance',counts)
    report.note(f"Baseline training contains {frame.height:,} candidate pairs; {int(frame['is_match'].sum()):,} are retained true matches. All candidates, including difficult retrieved negatives, were used; there was no negative subsampling or class reweighting.")
    summary=pl.DataFrame([{'feature':name,'minimum':frame[name].min(),'median':frame[name].median(),
        'maximum':frame[name].max()} for name in FEATURES])
    report.table('feature_summary',summary,limit=20)
    """),cell('md',"""
    ## Compare matching and nonmatching pairs

    Each histogram is normalized within its class, so the much larger negative class does not
    obscure positives. Cosine uses the same train-fitted TF-IDF vectors as retrieval. Jaccard
    measures token overlap; edit similarity uses unit-cost normalized Levenshtein distance.
    Length ratios preserve the shorter/longer relationship. Exact agreement requires both values
    present. Missingness is encoded separately; two missing addresses are not an exact match.
    Number conflict means both digit-token sets exist but differ; leading zeros are retained.
    """),cell('code',"""
    hist=pl.read_csv(RUN/'features/baseline/training_histograms.csv')
    report.table('feature_histograms',hist,limit=8)
    for name in ['name_cosine','address_cosine','name_edit_similarity','address_edit_similarity','name_token_jaccard','address_number_conflict']:
        fig=go.Figure()
        for label in [False,True]:
            part=hist.filter((pl.col('feature')==name)&(pl.col('is_match')==label))
            fig.add_bar(x=((part['bin_left']+part['bin_right'])/2).to_list(),y=part['fraction_within_class'].to_list(),
                name='True matches' if label else 'Nonmatches',opacity=.65)
        report.chart(name,fig,name.replace('_',' ').capitalize(), 'Feature value','Fraction within class',barmode='overlay')
    """),cell('md',"""
    ## Interpretation

    No single similarity is treated as proof of identity. A strong address can help when names
    differ; a similar name with conflicting numbers can still be a nonmatch. Characters from
    different scripts may share no ngrams. These plots describe the constructed training candidate
    pool, not the original population. Cleaning variants use the same feature definitions with
    their own frozen text views and training-fitted vocabulary/IDF.

    Continue with `05_model_comparison.ipynb` or the offline interactive walkthrough. The walkthrough
    also lets you inspect distributions of all 20 features across the three variants.
    """),cell('code',"""
    (report.folder/'report_manifest.json').write_text(json.dumps({'charts':report.figures,'tables':report.table_names,
        'scope':'training candidates only; saved features'},indent=2)+'\\n',encoding='utf-8')
    """)]


def model_notebook():
    return [cell('md',"""
    # Logistic regression versus XGBoost: matching candidate pairs

    Six pipelines: two fixed classifiers × three saved text/candidate variants. Both classifiers
    use the same 20 features and pairs within a variant. Logistic regression uses training-only
    standardization; XGBoost uses 200 depth-3 trees, with no early stopping or broad parameter search.

    Thresholds maximize the competition's **macro F0.5 per reference business**, including singleton
    credit. All distinct validation scores are tested, with tied scores accepted together. Unretrieved
    true pairs remain end-to-end false negatives. AP is candidate-conditional, not whole-pipeline recall.
    """),setup('model_comparison'),cell('code',"""
    comparison=pl.read_csv(RUN/'validation_comparison.csv')
    selection=json.loads((RUN/'selection.json').read_text(encoding='utf-8'))
    report.table('validation_comparison',comparison.select('pipeline_id','threshold','macro_f05','micro_precision','micro_recall',
        'exact_set_accuracy','singleton_correctness','candidate_average_precision','fit_seconds','inference_seconds'))
    fig=go.Figure()
    for model in ['logistic_regression','xgboost']:
        part=comparison.filter(pl.col('model')==model)
        fig.add_bar(x=part['variant'].to_list(),y=part['macro_f05'].to_list(),name=model)
    report.chart('model-comparison',fig,'Validation macro F0.5 by pipeline','Text / candidate variant','Macro F0.5',barmode='group')
    selected=selection['pipeline_id']
    report.note('Validation selected '+selected+' at threshold '+str(selection['threshold'])+'. The winner was frozen before any evaluation records were opened; the model was not refitted after selection.')
    """),cell('md',"""
    ## Threshold trade-off and model decisions

    The selected threshold is not assumed to be 0.5. Precision/recall here pool links, while macro
    F0.5 averages per business; these are intentionally different aggregations. The standalone
    walkthrough exposes a threshold slider for validation only. Its displayed grid metrics are
    exact at each shown threshold, and include the exact selected threshold.
    """),cell('code',"""
    curve=json.loads((RUN/'models'/selected/'display_curve.json').read_text(encoding='utf-8'))
    fig=go.Figure()
    for name in ['micro_precision','micro_recall','macro_f05','singleton_correctness']:
        fig.add_scatter(x=[r['threshold'] for r in curve],y=[r[name] for r in curve],mode='lines',name=name)
    fig.add_vline(x=selection['threshold'],line_dash='dot')
    report.chart('thresholds',fig,'Validation decision threshold','Threshold','Metric')
    errors=pl.read_csv(RUN/'models'/selected/'validation_errors.csv',infer_schema=False)
    counts=errors.group_by('error').len(name='pairs')
    report.table('validation_errors',errors,limit=12)
    report.table('validation_error_counts',counts)
    report.chart('validation-errors',go.Figure(go.Bar(x=counts['error'].to_list(),y=counts['pairs'].to_list())),
        'Validation errors by stage','Error type','Pairs')
    """),cell('md',"""
    ## Explaining scores

    Logistic contributions are standardized feature × coefficient; XGBoost contributions use its
    native SHAP API. Contributions plus the base term were checked against the raw model margin.
    These are explanations in **log-odds units**, not causal effects or calibrated probability changes.
    The next chart averages absolute contributions over validation candidates, which are mostly
    nonmatches. Use the walkthrough to inspect individual accepted/rejected pairs instead of
    interpreting this aggregate as a universal feature ranking.
    """),cell('code',"""
    contributions=np.load(RUN/'models'/selected/'validation_contributions.npz')['values']
    importance=np.abs(contributions[:,:-1]).mean(axis=0)
    order=np.argsort(importance)[-12:]
    fig=go.Figure(go.Bar(x=importance[order],y=[FEATURES[i] for i in order],orientation='h'))
    report.chart('contributions',fig,'Selected model: mean absolute validation contribution',
        'Mean absolute contribution (raw log-odds)','Feature',height=550)
    report.table('validation_slices',pl.read_csv(RUN/'models'/selected/'validation_slices.csv'),limit=18)
    """),cell('md',"""
    ## Cleaning effect: pipeline comparisons

    Reusing the same feature definitions does not hold candidate identities fixed. Cleaning affects
    both retrieval and features. Differences below are measured pipeline outcomes, not isolated
    causal effects of individual cleaning rules. Each pipeline has its own validation-selected threshold.
    """),cell('code',"""
    base={r['model']:r['macro_f05'] for r in comparison.filter(pl.col('variant')=='baseline').to_dicts()}
    deltas=comparison.with_columns(pl.Series('delta_macro_f05',[r['macro_f05']-base[r['model']] for r in comparison.to_dicts()]))
    report.table('cleaning_deltas',deltas.select('pipeline_id','delta_macro_f05'))
    fig=go.Figure()
    for model in ['logistic_regression','xgboost']:
        part=deltas.filter(pl.col('model')==model)
        fig.add_bar(x=part['variant'].to_list(),y=part['delta_macro_f05'].to_list(),name=model)
    report.chart('cleaning-deltas',fig,'Validation macro F0.5 change relative to each model baseline',
        'Variant','Change in macro F0.5',barmode='group')
    gap=float(comparison.filter(pl.col('pipeline_id')==selected)['macro_f05'][0])-base[selection['model']]
    report.note(f'The selected cleaned pipeline changes validation macro F0.5 by {gap:+.6f} relative to its model baseline. This small difference is not strong evidence of a general cleaning benefit; no significance claim is made.')
    """),cell('md',"""
    ## One frozen evaluation pass

    Only the selected pipeline was evaluated. Predictions were saved before evaluation labels were
    opened. The same train-fitted model, text rules, vectorizers, features, K=10, and selected threshold
    were used. No evaluation threshold search occurred. The confidence interval resamples Source 1
    businesses for this fixed pipeline; it excludes training/selection uncertainty.
    """),cell('code',"""
    evaluation=json.loads((RUN/'evaluation/metrics.json').read_text(encoding='utf-8'))
    report.table('evaluation_summary',pl.DataFrame([{name:evaluation[name] for name in
        ['macro_f05','micro_precision','micro_recall','micro_f1','exact_set_accuracy','singleton_count','singleton_correctness','tp','fp','fn']}]))
    report.table('evaluation_slices',pl.read_csv(RUN/'evaluation/slices.csv'),limit=18)
    stage_counts={'missed_by_blocking':evaluation['blocking']['missed_true_pairs'],
                  'missed_by_classifier':evaluation['fn']-evaluation['blocking']['missed_true_pairs'],
                  'false_merges':evaluation['fp']}
    report.chart('evaluation-errors',go.Figure(go.Bar(x=list(stage_counts),y=list(stage_counts.values()))),
        'Frozen evaluation error decomposition','Error type','Pairs')
    ci=evaluation['macro_f05_bootstrap_95']
    report.note(f"Evaluation macro F0.5 = {evaluation['macro_f05']:.6f}, bootstrap 95% interval [{ci['low']:.6f}, {ci['high']:.6f}]. Observed singleton correctness {evaluation['singleton_correctness']:.2%} is based on {evaluation['singleton_count']} cases, not a guarantee.")
    report.note('Evaluation is now consumed. Future model experiments must use training/validation; repeated tuning against this evaluation would remove its holdout interpretation. These reduced-pool results do not establish full competition or unseen-France performance.')
    (report.folder/'findings.md').write_text('# Model comparison findings\\n\\n'+'\\n'.join('- '+item for item in report.findings)+'\\n',encoding='utf-8')
    (report.folder/'report_manifest.json').write_text(json.dumps({'charts':report.figures,'tables':report.table_names,
        'selected_pipeline':selected,'evaluation_runs':1},indent=2)+'\\n',encoding='utf-8')
    display(Markdown('Open `reports/model_comparison/interactive_walkthrough.html` to explore saved validation examples, score contributions, and threshold choices.'))
    """),cell('md',"""
    References: [XGBoost](https://arxiv.org/abs/1603.02754),
    [tree feature attribution](https://arxiv.org/abs/1802.03888),
    [blocking survey](https://arxiv.org/abs/1905.06167).
    Full methodology and bibliography are in `docs/model_comparison.md` and `docs/references.bib`.
    """)]


def walkthrough():
    meta=json.loads((OUTPUT/'run.json').read_text(encoding='utf-8'))
    for relative,expected in meta['artifact_sha256'].items():
        if sha(OUTPUT/relative)!=expected:raise ValueError(f'Frozen model artifact changed: {relative}')
    selection=json.loads((OUTPUT/'selection.json').read_text(encoding='utf-8'))
    evaluation=json.loads((OUTPUT/'evaluation/metrics.json').read_text(encoding='utf-8'))
    comparison=pl.read_csv(OUTPUT/'validation_comparison.csv')
    sources={name:pl.read_csv(DATA/'validation'/f'{name}.tsv',separator='\t',infer_schema=False) for name in ('source1','source2','source3')}
    refs={row['entity_id']:row for row in sources['source1'].to_dicts()}
    targets={row['entity_id']:row for name in ('source2','source3') for row in sources[name].to_dicts()}
    target_countries={source:{country_key(value) for value in sources[source]['country']} - {''}
                      for source in ('source2','source3')}
    truth={key:set(value.split(',')) if value else set() for key,value in pl.read_csv(DATA/'validation/ground_truth.tsv',separator='\t',infer_schema=False).iter_rows()}
    choices=[]
    def add(key):
        if key not in choices and len(choices)<20:choices.append(key)
    chosen_scores=pl.read_parquet(OUTPUT/'models'/selection['pipeline_id']/'validation_predictions.parquet')
    chosen_predictions=predictions_at(chosen_scores,chosen_scores['score'].to_numpy(),truth,selection['threshold'])
    for country in sorted(sources['source1']['country'].unique().to_list()):
        for key in sources['source1'].filter(pl.col('country')==country).sort('entity_id')['entity_id']:
            if truth[key] and chosen_predictions[key]==truth[key]:add(key);break
    for pipeline in [selection['pipeline_id'],*comparison['pipeline_id'].to_list()]:
        errors=pl.read_csv(OUTPUT/'models'/pipeline/'validation_errors.csv',infer_schema=False)
        for kind in ('false_merge','missed_by_blocking','missed_by_classifier'):
            for key in errors.filter(pl.col('error')==kind)['source1_entity_id'].unique(maintain_order=True).head(1):add(key)
    for key in sorted(truth):
        if not truth[key]:add(key);break
    for country in sorted(sources['source1']['country'].unique().to_list()):
        for key in sources['source1'].filter(pl.col('country')==country).sort('entity_id')['entity_id'].head(2):add(key)
    pipelines=[];histograms={};cases={key:{'id':key,'record':refs[key],'truth':sorted(truth[key]),'variants':{}} for key in choices}
    training_documents=sum(pl.read_csv(DATA/'train'/f'{source}.tsv',separator='\t',infer_schema=False,
                                       columns=['entity_id']).height for source in ('source1','source2','source3'))
    for variant in VARIANTS:
        print(f'Building saved weight explanations: {variant}',flush=True)
        train=pl.read_parquet(OUTPUT/'features'/variant/'train.parquet')
        train_matrix=train.select(FEATURES).to_numpy();labels=train['is_match'].to_numpy().astype(int)
        with (PROJECT/'artifacts/cleaning_v1/validation'/variant/'tfidf_vectorizers.pkl').open('rb') as stream:
            vectorizers=pickle.load(stream)
        features=pl.read_parquet(OUTPUT/'features'/variant/'validation.parquet')
        lookup={pair:i for i,pair in enumerate(features.select('source1_entity_id','target_id').iter_rows())}
        model_data={}
        for name in MODEL_NAMES:
            pid=variant+'__'+name;folder=OUTPUT/'models'/pid
            metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
            scored=pl.read_parquet(folder/'validation_predictions.parquet')
            assert scored.select('source1_entity_id','target_id').equals(features.select('source1_entity_id','target_id'))
            model_data[name]={'scores':scored['score'].to_numpy(),
                              'contributions':np.load(folder/'validation_contributions.npz')['values']}
            with (folder/'model.pkl').open('rb') as stream:model=pickle.load(stream)
            with threadpool_limits(limits=2):training=model_weights(model,name,train_matrix,labels)
            model_data[name]['training']=training
            variant_label={'baseline':'Current normalization','placeholders':'Placeholder cleanup','placeholders_abbreviations':'Cleanup + abbreviations'}[variant]
            model_label={'logistic_regression':'Logistic regression','xgboost':'XGBoost'}[name]
            pipelines.append({'pipeline_id':pid,'variant':variant,'model':name,'label':variant_label+' · '+model_label,
                'variant_label':variant_label,'model_label':model_label,
                'threshold':metrics['threshold'],'metrics':metrics,'training':training,
                'curve':json.loads((folder/'display_curve.json').read_text(encoding='utf-8'))})
        histograms[variant]=pl.read_csv(OUTPUT/'features'/variant/'training_histograms.csv').to_dicts()
        evidence=pl.read_parquet(PROJECT/'artifacts/cleaning_v1/validation'/variant/'retrieval_evidence.parquet').filter(pl.col('source1_entity_id').is_in(choices))
        ranks={}
        for row in evidence.to_dicts():ranks.setdefault((row['source1_entity_id'],row['target_id']),{})[row['view']]=row['rank']
        for key in choices:
            rows=[]
            for pair in features.filter(pl.col('source1_entity_id')==key).to_dicts():
                target=pair['target_id'];i=lookup[(key,target)]
                weights={view:tfidf_details(vectorizers[view],
                    clean_for_retrieval(refs[key][field],field,variant),clean_for_retrieval(targets[target][field],field,variant),
                    training_documents) for view,field in [('name','business_name'),('address','business_address')]}
                for view,index in [('name',0),('address',10)]:
                    if abs(weights[view]['cosine']-pair[FEATURES[index]])>2e-6:raise ValueError('TF-IDF explanation/cosine mismatch')
                feature_values=[float(pair[feature]) for feature in FEATURES]
                tree_model=model_data['xgboost']['training']
                leaves=[tree_path(tree,feature_values)[1]['leaf'] for tree in tree_model['trees']]
                if abs(float(sigmoid(tree_model['base_margin']+sum(leaves)))-model_data['xgboost']['scores'][i])>3e-6:
                    raise ValueError('Dumped tree leaves do not reconstruct saved XGBoost score')
                rows.append({'target_id':target,'record':targets[target],'is_match':pair['is_match'],
                    'features':feature_values,'tfidf':weights,'ranks':ranks[(key,target)],
                    'scores':{name:float(model_data[name]['scores'][i]) for name in MODEL_NAMES},
                    'contributions':{name:np.round(model_data[name]['contributions'][i],7).tolist() for name in MODEL_NAMES}})
            rows.sort(key=lambda row:(-row['scores'][selection['model']],row['target_id']))
            eligible={source:sum(eligible_country(country_key(refs[key]['country']),country_key(value),
                target_countries[source]) for value in sources[source]['country'])
                for source in ('source2','source3')}
            retrieval_lists={source:{view:evidence.filter((pl.col('source1_entity_id')==key)&
                (pl.col('target_source')==source)&(pl.col('view')==view)).sort('rank').select('target_id','rank','cosine').to_dicts()
                for view in ('name','address')} for source in ('source2','source3')}
            cases[key]['variants'][variant]={'pairs':rows,'bucketing':{
                'all_targets':len(targets),'eligible_by_source':eligible,'top_k':10,
                'lists':retrieval_lists,'training_documents':training_documents,
                'vocabulary_sizes':{view:len(v.vocabulary_) for view,v in vectorizers.items()}},'reference_view':{
                'name':clean_for_retrieval(refs[key]['business_name'],'business_name',variant),
                'address':clean_for_retrieval(refs[key]['business_address'],'business_address',variant)}}
    payload={'features':FEATURES,'pipelines':pipelines,'cases':list(cases.values()),'histograms':histograms,
        'selection':{name:selection[name] for name in ('pipeline_id','variant','model','threshold')},'evaluation':evaluation}
    folder=PROJECT/'reports/model_comparison'
    encoded=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    (folder/'walkthrough_data.json').write_text(encoded,encoding='utf-8')
    template=(PROJECT/'scripts/walkthrough.html').read_text(encoding='utf-8')
    learning=(PROJECT/'scripts/walkthrough_learning.js').read_text(encoding='utf-8')
    (folder/'interactive_walkthrough.html').write_text(template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',encoded)
        .replace('__LEARNING__',learning),encoding='utf-8')
    print(f'Interactive walkthrough: {len(cases)} validation examples, all six pipelines.',flush=True)
    from build_side_by_side import build
    build()


def main():
    meta=json.loads((OUTPUT/'run.json').read_text(encoding='utf-8'))
    if meta['status']!='complete' or (OUTPUT/'INCOMPLETE.txt').exists():raise ValueError('Model experiment is incomplete')
    for relative,expected in meta['artifact_sha256'].items():
        if sha(OUTPUT/relative)!=expected:raise ValueError(f'Model artifact changed: {relative}')
    os.environ['JUPYTER_RUNTIME_DIR']=str(PROJECT/'.jupyter_runtime')
    execute('04_pair_features','features',feature_notebook())
    execute('05_model_comparison','model_comparison',model_notebook())
    walkthrough()


if __name__=='__main__':main()
