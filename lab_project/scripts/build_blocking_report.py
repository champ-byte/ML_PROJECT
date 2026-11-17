"""Execute and export a notebook explaining the saved blocking experiment."""

import argparse
import os
from pathlib import Path
import sys
from textwrap import dedent

from bs4 import BeautifulSoup
from jupyter_client import KernelManager
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

PROJECT = Path(__file__).resolve().parents[1]


def build_notebook(run_dir):
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip()))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(dedent(text).strip()))

    md("""
    # Candidate generation: keeping true matches while reducing comparisons

    This notebook compares **exact-name blocking** with **character-ngram TF-IDF retrieval**.
    It explains the saved, bounded experiment at K=10,25,50. Character ngrams are short overlapping
    text fragments; TF-IDF weights common fragments less and cosine compares the resulting vectors.
    Name and address searches are independent; their **union** becomes the candidate set.

    **Two different jobs:** the blocker finds plausible alternatives; the future classifier decides
    which alternatives to accept. No classifier or final match predictions exist in this report.
    Ground truth is a post-hoc evaluation overlay and never inserts candidates into retrieval.

    Vocabulary/IDF are fitted on training records only. Validation chooses candidate cost versus
    recall. The held-out evaluation split and full competition test set remain unopened.
    These are measurements on the reduced candidate pool, not competition-scale estimates.
    """)
    code(f"""
    import hashlib
    import json
    from pathlib import Path
    import sys
    import polars as pl
    import plotly.graph_objects as go
    from IPython.display import Markdown, display

    PROJECT = Path({str(PROJECT)!r})
    # Recover the project relative to cwd if this notebook was moved with the repository.
    if not (PROJECT / 'scripts/blocking.py').exists():
        PROJECT = next((p if (p/'scripts/blocking.py').exists() else p/'lab_project')
                       for p in [Path.cwd(), *Path.cwd().parents]
                       if (p/'scripts/blocking.py').exists() or (p/'lab_project/scripts/blocking.py').exists())
    RUN = Path({str(run_dir)!r})
    if not RUN.exists():
        RUN = PROJECT / 'artifacts/blocking_v1'
    DATA = PROJECT / 'data/dataset_small_10k'
    sys.path.insert(0,str(PROJECT/'scripts'))
    from eda_helpers import Report, bars
    from blocking import normalize_text
    report = Report(PROJECT/'reports/blocking')
    run = json.loads((RUN/'run.json').read_text(encoding='utf-8'))
    assert run['status']=='complete' and not (RUN/'INCOMPLETE.txt').exists()
    assert not run['evaluation_opened'] and not run['classifier_trained']
    for relative, expected in run['input_hashes'].items():
        assert relative.split('/')[0] in ['train','validation']
        assert hashlib.sha256((DATA/relative).read_bytes()).hexdigest()==expected
    for relative,expected in run['artifact_sha256'].items():
        assert hashlib.sha256((RUN/relative).read_bytes()).hexdigest()==expected
    metrics=pl.read_csv(RUN/'validation_metrics.csv')
    slices=pl.read_csv(RUN/'validation_slices.csv')
    method=run['selected_method'] or run['diagnostic_method']
    selected=metrics.filter(pl.col('method')==method).to_dicts()[0]
    diagnostic_method=run['diagnostic_method']
    diagnostic=metrics.filter(pl.col('method')==diagnostic_method).to_dicts()[0]
    print('Input and artifact hashes verified. Report uses saved validation results.')
    """)
    md("""
    ## 1. What changed from raw text to retrieval text?

    The raw dataset is preserved. Derived views apply Unicode NFKC, case folding, punctuation/symbol
    spacing, and whitespace collapsing. Accents and non-Latin letters remain. There is no external
    address lookup, transliteration service, or business database.

    A country label is a dynamic partition, not a list hard-coded to US/India. Blank query countries
    or labels absent from a target source search that source globally; blank target countries are
    always eligible. A new country present in the target source uses its own partition.
    """)
    code("""
    sources={name:pl.read_csv(DATA/'validation'/f'{name}.tsv',separator='\\t',infer_schema=False)
             for name in ['source1','source2','source3']}
    targets=pl.concat([sources[name].with_columns(pl.lit(name).alias('source')) for name in ['source2','source3']])
    truth={key:set(value.split(',')) if value else set()
           for key,value in pl.read_csv(DATA/'validation/ground_truth.tsv',separator='\\t',infer_schema=False).iter_rows()}
    sample=sources['source1'].sort('entity_id').head(6)
    report.table('normalization_examples',sample.with_columns(
        pl.Series('retrieval_name',[normalize_text(value) for value in sample['business_name']]),
        pl.Series('retrieval_address',[normalize_text(value) for value in sample['business_address']])))
    report.note('No learned matching model is used. Matching countries and lexical overlap are retrieval heuristics with measured failure cases, not proof of shared identity.')
    """)
    md("""
    ### Inside one TF-IDF name vector

    The next chart shows the twelve largest normalized TF-IDF weights for the first example name.
    Each label is a 3–5 character fragment; `␠` denotes a space. Weights combine fragment frequency
    with its inverse frequency in the training corpus, then normalize the vector to unit length.
    They are text-representation weights, not probabilities or learned matching-model importance.
    """)
    code("""
    import pickle
    # Load only this experiment's own hash-verified fitted vectorizers.
    with (RUN/'tfidf_vectorizers.pkl').open('rb') as stream:
        vectorizers=pickle.load(stream)
    text=normalize_text(sample['business_name'][0])
    vector=vectorizers['name'].transform([text]).tocsr()
    vocabulary=vectorizers['name'].get_feature_names_out()
    weighted=sorted(zip(vector.indices,vector.data),key=lambda item:(-item[1],vocabulary[item[0]]))[:12]
    weights=pl.DataFrame({'fragment':[vocabulary[i] for i,value in weighted],
                         'tfidf_weight':[float(value) for i,value in weighted]})
    report.table('example_tfidf_weights',weights)
    fig=go.Figure(go.Bar(x=weights['tfidf_weight'].to_list(),
                        y=[value.replace(' ','␠') for value in weights['fragment']],orientation='h'))
    report.chart('tfidf-weights',fig,f'Top character fragments in: {text}',
                 'L2-normalized TF-IDF weight','Character fragment')
    """)
    md("""
    ## 2. How many true matches survive?

    **Blocking recall** = retained labeled pairs / all labeled pairs. **Complete-match coverage**
    counts nonsingleton businesses whose entire match list survived. Both matter when a business
    has multiple matching records. K is **per view, per source**; a union can contain up to 4K
    candidates. No extra cap is applied after the union.
    """)
    code("""
    report.table('validation_metrics',metrics.select('method','candidate_pairs','pair_recall','complete_match_coverage','pair_quality','reduction_ratio','mean_candidates','p95_candidates','zero_candidate_rate','oracle_macro_f05'),limit=12)
    fig=go.Figure()
    for view in ['name','address','union']:
        part=metrics.filter(pl.col('view')==view).sort('k')
        fig.add_scatter(x=part['k'].to_list(),y=part['pair_recall'].to_list(),mode='lines+markers',name=view)
    exact=metrics.filter(pl.col('method')=='exact_name').to_dicts()[0]
    fig.add_hline(y=exact['pair_recall'],line_dash='dot',annotation_text='Exact-name baseline')
    fig.add_hline(y=.98,line_dash='dash',line_color='#059669',annotation_text='Proposed 98% target')
    report.chart('recall-k',fig,'Validation blocking recall vs K','K per view / target source','Fraction of true pairs retained')
    fig=go.Figure(go.Scatter(x=metrics['candidate_pairs'].to_list(),y=metrics['pair_recall'].to_list(),
        mode='markers+text',text=metrics['method'].to_list(),textposition='top center',
        marker=dict(size=11),hovertemplate='%{text}<br>Pairs: %{x:,}<br>Recall: %{y:.3%}<extra></extra>'))
    fig.add_hline(y=.98,line_dash='dash')
    report.chart('recall-cost',fig,'Recall versus comparisons sent to the next phase','Candidate pairs','Blocking recall',height=540)
    report.note(f"Exact-name recall is {exact['pair_recall']:.2%}; {diagnostic_method} retains {diagnostic['pair_recall']:.2%} of true pairs with {diagnostic['candidate_pairs']:,} candidates.")
    """)
    md("""
    ## 3. What is the cost, and how uneven are the candidate lists?

    Reduction ratio uses the entire validation cross-source comparison space **N1 × (N2+N3)**,
    before country partitioning. Retrieval time is measured independently for each K. TF-IDF fit
    and validation transforms are common setup costs, reported separately. Name/address ablations
    share the union's retrieval run; their timing is not an independent single-view benchmark.
    Runtime is a single-machine observation, not a reliable ranking of small timing differences.
    """)
    code("""
    timed=metrics.filter(pl.col('view').is_in(['exact','union']))
    report.table('runtime',timed.select('method','retrieval_seconds','runtime_scope'))
    fig=go.Figure(go.Bar(x=timed['method'].to_list(),y=timed['retrieval_seconds'].to_list()))
    report.chart('runtime',fig,'Measured validation candidate-generation time','Method','Seconds; common TF-IDF setup excluded')
    per_business=pl.read_csv(RUN/'validation'/method/'per_business.csv')
    baseline_counts=pl.read_csv(RUN/'validation/exact_name/per_business.csv')
    fig=go.Figure()
    fig.add_histogram(x=baseline_counts['candidates'].to_list(),name='exact_name',opacity=.65)
    fig.add_histogram(x=per_business['candidates'].to_list(),name=method,opacity=.65)
    report.chart('candidate-counts',fig,'Candidate-list sizes by reference business','Candidate records','Reference businesses',barmode='overlay')
    report.note(f"Common TF-IDF fit: {run['fit_seconds']:.2f}s; validation preparation: {run['validation_prepare_seconds']:.2f}s. {method} reduces {selected['full_comparison_space']:,} possible comparisons by {selected['reduction_ratio']:.3%}. Peak memory was not measured.")
    """)
    md("""
    ## 4. Where does blocking miss true matches?

    Country/source slices show denominators so a small group is not mistaken for a precise estimate.
    For missed labeled pairs, diagnostics distinguish country exclusion, zero query vectors,
    no shared character ngrams, and positive overlap falling outside top K. Diagnostics do not
    add missed pairs to the actual candidate outputs.
    """)
    code("""
    part=slices.filter(pl.col('method')==method)
    report.table('coverage_slices',part.select('slice_type','slice','true_pairs','retained_true_pairs','pair_recall','complete_match_coverage'))
    fig=go.Figure(go.Bar(x=[f"{row['slice_type']}: {row['slice']}" for row in part.to_dicts()],y=part['pair_recall'].to_list()))
    report.chart('coverage-slices',fig,f'Coverage slices: {method}','Slice','Blocking recall')
    misses=pl.read_csv(RUN/'missed_true_pairs.csv',infer_schema=False)
    report.table('missed_true_pairs',misses,limit=10)
    reasons=misses.group_by('reason').len(name='missed_pairs').sort('reason')
    report.table('miss_reasons',reasons)
    fig=go.Figure(go.Bar(x=reasons['reason'].to_list(),y=reasons['missed_pairs'].to_list()))
    report.chart('miss-reasons',fig,f'Why labeled pairs were missed: {diagnostic_method}','Diagnostic reason','Missed true pairs')
    fig=go.Figure(go.Sankey(node=dict(label=['All true validation pairs','Retained by blocker','Missed by blocker']),
                           link=dict(source=[0,0],target=[1,2],value=[diagnostic['retained_true_pairs'],diagnostic['missed_true_pairs'] ],color=['#86efac','#fca5a5'])))
    report.chart('true-pair-flow',fig,'True pairs retained and lost before classification')
    report.note(f"{diagnostic['missed_true_pairs']:,} true pairs are absent from {diagnostic_method}. They must remain false negatives in end-to-end evaluation, even if a future classifier is perfect on its candidates.")
    """)
    md("""
    ## 5. Inspect individual candidate lists

    Choose a reference business in the dropdown. Each point is a retrieved target record; hover
    for its text and view-specific ranks. Green points are known true matches; gray points are
    nonmatches according to the provided labels. This coloring is a **post-hoc truth overlay**.
    The x-axis is the larger name/address cosine, not a learned match probability or a final decision.

    For legibility the chart shows the eight highest-scoring candidates plus all retained true
    matches. Exported TSV files and Parquet evidence retain the full candidate sets.
    """)
    code("""
    evidence=pl.read_parquet(RUN/'validation'/diagnostic_method/'retrieval_evidence.parquet')
    selected_businesses=pl.read_csv(RUN/'validation'/diagnostic_method/'per_business.csv')
    choices=[]
    # Reproducible diagnostic examples: a miss, singleton, complete recovery, then country coverage.
    for subset in [selected_businesses.filter(pl.col('missed')>0),
                   selected_businesses.filter((pl.col('true_matches')==0)&(pl.col('candidates')>0)),
                   selected_businesses.filter((pl.col('true_matches')>0)&(pl.col('missed')==0))]:
        for key in subset.sort('source1_entity_id')['source1_entity_id'].head(1):
            if key not in choices: choices.append(key)
    for country in sorted(sources['source1']['country'].unique().to_list()):
        for key in sources['source1'].filter(pl.col('country')==country).sort('entity_id')['entity_id'].head(2):
            if key not in choices: choices.append(key)
    choices=choices[:6]
    refs={row['entity_id']:row for row in sources['source1'].to_dicts()}
    target_lookup={row['entity_id']:row for row in targets.to_dicts()}
    report.table('explorer_references',sources['source1'].filter(pl.col('entity_id').is_in(choices)))
    fig=go.Figure(); buttons=[]
    for index,key in enumerate(choices):
        grouped={}
        for row in evidence.filter(pl.col('source1_entity_id')==key).to_dicts():
            item=grouped.setdefault(row['target_id'],{'best':0.,'name_rank':'—','address_rank':'—'})
            item['best']=max(item['best'],row['cosine']);item[row['view']+'_rank']=row['rank']
        ordered=sorted(grouped,key=lambda target:(-grouped[target]['best'],target))
        shown=[target for target in ordered if target in set(ordered[:8])|truth[key]]
        custom=[[target_lookup[target]['business_name'],target_lookup[target]['business_address'] or '',
                 grouped[target]['name_rank'],grouped[target]['address_rank'],target in truth[key]] for target in shown]
        fig.add_scatter(x=[grouped[target]['best'] for target in shown],y=shown,mode='markers',name=key,visible=index==0,
            marker=dict(size=12,color=['#059669' if target in truth[key] else '#94a3b8' for target in shown]),customdata=custom,
            hovertemplate='%{y}<br>%{customdata[0]}<br>%{customdata[1]}<br>Name rank: %{customdata[2]} | Address rank: %{customdata[3]}<br>Labeled true match: %{customdata[4]}<br>Max cosine: %{x:.4f}<extra></extra>')
        buttons.append(dict(label=key+' / '+refs[key]['country'],method='update',args=[{'visible':[i==index for i in range(len(choices))]}, {'title':f"Candidates for {key}: {refs[key]['business_name']}"}]))
    report.chart('candidate-explorer',fig,f"Candidates for {choices[0]}: {refs[choices[0]]['business_name']}",
                 'Maximum cosine across retrieval views (not match probability)','Candidate ID',height=620,
                 updatemenus=[dict(buttons=buttons,x=1,y=1.17,xanchor='right')],showlegend=False)
    """)
    md("""
    ## 6. A small candidate graph

    Edges join one reference business to a few candidate target records. Solid edges are actual
    retrieved candidates; dashed red edges are **unretrieved true pairs added only to this diagnostic
    drawing**. These dashed edges are not in the candidate TSV or the inference set.
    """)
    code("""
    key=choices[0]
    evidence_rows=evidence.filter(pl.col('source1_entity_id')==key).sort(['rank','target_id'])
    candidate_ids=list(dict.fromkeys(evidence_rows['target_id'].to_list()))
    shown=list(dict.fromkeys(candidate_ids[:8]+sorted(truth[key])))
    retained=set(candidate_ids); fig=go.Figure()
    for index,target in enumerate(shown):
        is_true=target in truth[key];is_candidate=target in retained
        color='#059669' if is_true and is_candidate else '#dc2626' if is_true else '#94a3b8'
        fig.add_scatter(x=[0,1],y=[.5,index/max(1,len(shown)-1)],mode='lines',line=dict(color=color,dash='solid' if is_candidate else 'dash'),showlegend=False,
                        hovertemplate=f'{key} → {target}<br>Retrieved: {is_candidate}<br>Labeled true: {is_true}<extra></extra>')
    fig.add_scatter(x=[0],y=[.5],mode='markers+text',text=[key],textposition='top center',marker=dict(size=20,color='#2563eb'),showlegend=False)
    fig.add_scatter(x=[1]*len(shown),y=[i/max(1,len(shown)-1) for i in range(len(shown))],mode='markers+text',text=shown,textposition='middle right',marker=dict(size=12,color='#334155'),showlegend=False)
    fig.update_xaxes(visible=False,range=[-.2,1.6]);fig.update_yaxes(visible=False)
    report.chart('candidate-graph',fig,f'Small diagnostic graph for {key}',height=560)
    """)
    md("""
    ## 7. Decision and limits

    We select the lowest candidate count among exact-name/union methods meeting 98% validation
    pair recall, then use retrieval time as a tie-breaker. The 98% line is an engineering target,
    not a statistical guarantee. If no method meets it, this bounded experiment stops before
    classifier work. Training candidates are exported only if a configuration meets the target.

    **Oracle macro F0.5** assumes a perfect future classifier that keeps only true candidates,
    including correct empty sets for singletons. It is an upper bound for this candidate set,
    not an achieved model score. Final precision, recall, and F0.5 require actual matching decisions.

    References: [blocking survey](https://arxiv.org/abs/1905.06167),
    [blocking versus nearest-neighbor comparison](https://arxiv.org/abs/2202.12521).
    Our methods are proposed adaptations; these papers do not establish results on this dataset.
    """)
    code("""
    decision=(f"Selected {run['selected_method']} for the next phase." if run['recall_target_met'] else
              'No tested configuration meets 98% validation recall. Investigate the recorded misses before classifier tuning.')
    report.note(decision)
    report.note(f"{method}: oracle macro F0.5 = {selected['oracle_macro_f05']:.4f}; pair quality = {selected['pair_quality']:.4%}. Pair quality is the true-match fraction in candidates, not a trained model's precision.")
    if run['training_summary']:
        train=run['training_summary']
        report.note(f"Exported {train['candidate_pairs']:,} training candidate pairs for {run['selected_method']}. Training recall {train['pair_recall']:.2%} is an in-sample diagnostic, not held-out performance.")
    report.note('No evaluation records opened and no classifier trained. Candidate generation on the full competition pool remains an unmeasured scaling/generalization question.')
    findings='# Candidate-generation findings\\n\\n'+ '\\n'.join('- '+item for item in report.findings)
    (report.folder/'findings.md').write_text(findings+'\\n',encoding='utf-8')
    (report.folder/'report_manifest.json').write_text(json.dumps({'run_dir':str(RUN),'scope':'train fit / validation selection only',
        'charts':report.figures,'tables':report.table_names,'classifier_trained':False,'evaluation_opened':False},indent=2)+'\\n',encoding='utf-8')
    display(Markdown('Candidate sets, per-view rank/score evidence, metrics, hashes, and the fit-only vectorizers are saved under `artifacts/blocking_v1/`. The original records remain unchanged.'))
    """)
    return nbformat.v4.new_notebook(cells=cells,metadata={
        "kernelspec":{"display_name":"Python 3 (ML lab uv environment)","language":"python","name":"python3"},
        "language_info":{"name":"python","version":sys.version.split()[0]}})


def main(run_dir):
    notebook=build_notebook(Path(run_dir).resolve())
    path=PROJECT/'notebooks/02_candidate_generation.ipynb'
    folder=PROJECT/'reports/blocking'
    folder.mkdir(parents=True,exist_ok=True)
    path.parent.mkdir(exist_ok=True)
    os.environ['JUPYTER_RUNTIME_DIR']=str(PROJECT/'.jupyter_runtime')
    nbformat.write(notebook,path)
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    try:
        NotebookClient(notebook,km=manager,timeout=180,startup_timeout=60,
                       resources={'metadata':{'path':str(PROJECT)}}).execute()
    finally:
        if manager.has_kernel: manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook,path)
    exporter=HTMLExporter();exporter.exclude_input=True
    body,_=exporter.from_notebook_node(notebook)
    document=BeautifulSoup(body,'html.parser')
    for script in document.find_all('script',src=True): script.decompose()
    (folder/'candidate_generation.html').write_text(str(document),encoding='utf-8')
    print(f'Executed notebook: {path}\nOffline report: {folder / "candidate_generation.html"}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'artifacts/blocking_v1')
    main(parser.parse_args().run_dir)
