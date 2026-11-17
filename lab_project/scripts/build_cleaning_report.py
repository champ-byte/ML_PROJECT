"""Author, execute, and export the controlled cleaning/blocking comparison notebook."""

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

PROJECT=Path(__file__).resolve().parents[1]


def build_notebook(run_dir):
    cells=[]

    def md(text):cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip()))
    def code(text):cells.append(nbformat.v4.new_code_cell(dedent(text).strip()))

    md("""
    # Does conservative data cleaning improve candidate generation?

    **Controlled experiment:** preserve the saved baseline; change derived retrieval text only.
    The three variants are current normalization, field-aware placeholder removal, and placeholder
    removal plus a small field-specific abbreviation map.

    **Flow:** original records → field-aware cleanup → existing normalization → training-only
    TF-IDF fit → fixed K=10 retrieval → paired validation comparison.

    Character ngrams (3–5), cosine retrieval, country partitions, tie handling, and K stay fixed.
    Vocabulary/IDF are refitted on training records for each variant, so the measured effect
    includes both cleaned text and its resulting representation. It is not an experiment that
    holds the old vocabulary fixed. Audit/rules were saved before opening validation records.

    No raw files or baseline artifacts are changed. No classifier is trained; evaluation and
    competition test records remain unopened. Validation is used for comparison and therefore
    is not an independent final performance estimate.
    """)
    code(f"""
    import hashlib
    import json
    from pathlib import Path
    import sys
    from html import escape
    import polars as pl
    import plotly.graph_objects as go
    from IPython.display import display, Markdown

    PROJECT=Path({str(PROJECT)!r})
    if not (PROJECT/'scripts/cleaning.py').exists():
        PROJECT=next((p if (p/'scripts/cleaning.py').exists() else p/'lab_project')
            for p in [Path.cwd(),*Path.cwd().parents]
            if (p/'scripts/cleaning.py').exists() or (p/'lab_project/scripts/cleaning.py').exists())
    RUN=Path({str(run_dir)!r})
    if not RUN.exists():RUN=PROJECT/'artifacts/cleaning_v1'
    DATA=PROJECT/'data/dataset_small_10k'
    sys.path.insert(0,str(PROJECT/'scripts'))
    from eda_helpers import Report
    report=Report(PROJECT/'reports/cleaning')
    run=json.loads((RUN/'run.json').read_text(encoding='utf-8'))
    assert run['status']=='complete' and not (RUN/'INCOMPLETE.txt').exists()
    assert not run['classifier_trained'] and not run['evaluation_opened']
    assert run['baseline_reproduced_exactly']
    for relative,expected in run['input_hashes'].items():
        assert relative.split('/')[0] in ['train','validation']
        assert hashlib.sha256((DATA/relative).read_bytes()).hexdigest()==expected
    for relative,expected in run['artifact_sha256'].items():
        assert hashlib.sha256((RUN/relative).read_bytes()).hexdigest()==expected
    rules=json.loads((RUN/'rules.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((RUN/'rules.json').read_bytes()).hexdigest()==run['protocol']['rules_sha256']
    metrics=pl.read_csv(RUN/'validation_comparison.csv')
    variants=run['protocol']['variants']
    colors={{'baseline':'#2563eb','placeholders':'#059669','placeholders_abbreviations':'#d97706'}}
    audit=pl.read_csv(RUN/'training_audit/counts.csv')
    examples=pl.read_csv(RUN/'training_audit/examples.csv',infer_schema=False)
    print('Verified frozen rules, original inputs, saved experiment artifacts, and exact baseline reproduction.')
    """)
    md("""
    ## 1. What did the training-only audit find?

    Count affected fields, not unique businesses. A field can have several issues, so bars are not
    mutually exclusive. Extra whitespace is already handled by the baseline. A Unicode format
    character is not automatically an error: non-joiners may matter in Indic names and are retained.
    """)
    code("""
    report.table('training_audit_counts',audit,limit=18)
    totals=audit.group_by('issue').agg(pl.col('affected_records').sum()).sort('issue')
    fig=go.Figure(go.Bar(x=totals['issue'].to_list(),y=totals['affected_records'].to_list()))
    report.chart('audit-issues',fig,'Training fields affected by each audit indicator','Indicator (overlap permitted)','Affected field instances',height=550)
    components=pl.read_csv(RUN/'training_audit/address_components.csv')
    protected=pl.read_csv(RUN/'training_audit/protected_names.csv',infer_schema=False)
    formats=pl.read_csv(RUN/'training_audit/format_characters.csv',infer_schema=False)
    report.table('placeholder_components',components)
    report.table('protected_names',protected)
    report.table('preserved_format_characters',formats)
    placeholder_count=int(totals.filter(pl.col('issue')=='placeholder_cleanup_changes_view')['affected_records'][0])
    report.note(f"Placeholder cleanup changes {placeholder_count:,} training field views. The address audit found {components['occurrences'].sum():,} placeholder components. {protected.height} names containing Null or N/A are retained as business text.")
    """)
    md("""
    ## 2. Which rules are actually applied?

    **Placeholder cleanup:** remove exact comma-delimited address components `NULL`, `null`, `N/A`,
    and `<NULL>`; remove explicit embedded `<NULL>` in addresses and `<CITY_NAME>` in names.
    Bare words in real names or inside useful address components are not deleted.

    **Additional abbreviation variant:** expand `pvt/ltd/inc/corp` in names and
    `rd/ave/blvd/apt/ste` in addresses. The mappings are proposed heuristics to measure, not claims
    of universal equivalence. `St` (Street or Saint), `No`, transliteration, typo correction, and
    deletion of legal suffixes are excluded. Country partitions are unchanged.
    """)
    code("""
    mapping=pl.DataFrame([{'field':field,'abbreviation':key,'expansion':value}
        for field,label in [('business_name','name_abbreviations'),('business_address','address_abbreviations')]
        for key,value in rules[label].items()])
    report.table('abbreviation_map',mapping)
    changed_examples=examples.filter(pl.col('issue').is_in(['placeholder_cleanup_changes_view','abbreviation_expansion_changes_view']))
    display_examples=changed_examples.group_by('issue',maintain_order=True).head(6)
    report.table('training_before_after',display_examples,limit=16)
    choices=changed_examples.group_by('issue',maintain_order=True).head(4).unique(subset=['entity_id','field'],maintain_order=True).to_dicts()
    fig=go.Figure();buttons=[]
    for index,row in enumerate(choices):
        values=[row['raw'],row['baseline'],row['placeholders'],row['placeholders_abbreviations']]
        fig.add_trace(go.Table(visible=index==0,columnwidth=[1,1,1,1],
            header=dict(values=['Raw text','Baseline view','Placeholder cleanup','+ Abbreviations'],fill_color='#e2e8f0',align='left'),
            cells=dict(values=[[escape(value or '')] for value in values],align='left',height=120,font=dict(size=13))))
        buttons.append(dict(label=row['entity_id']+' / '+row['field'].replace('business_',''),method='update',
            args=[{'visible':[i==index for i in range(len(choices))]},{'title':f"Training example: {row['entity_id']} / {row['field']}"}]))
    report.chart('text-examples',fig,f"Training example: {choices[0]['entity_id']} / {choices[0]['field']}",
                 height=400,updatemenus=[dict(buttons=buttons,x=1,y=1.2,xanchor='right')])
    """)
    md("""
    ## 3. Does cleaning change validation coverage and cost?

    Blocking recall measures retained true pairs. Pair quality measures the true-match fraction
    of the candidate pool; it is **not** precision from a trained matcher. Complete coverage is
    calculated over nonsingletons. Oracle F0.5 assumes a perfect future matcher and is only a ceiling.
    """)
    code("""
    report.table('validation_comparison',metrics.select('variant','retained_true_pairs','missed_true_pairs','pair_recall',
        'complete_match_coverage','candidate_pairs','pair_quality','mean_candidates','oracle_macro_f05'))
    fig=go.Figure()
    for variant in variants:
        row=metrics.filter(pl.col('variant')==variant).to_dicts()[0]
        fig.add_scatter(x=[row['candidate_pairs']],y=[100*row['pair_recall']],mode='markers+text',
            text=[variant],name=variant,textposition='top center',marker=dict(size=14,color=colors[variant]),
            hovertemplate='%{text}<br>Candidates: %{x:,}<br>Blocking recall: %{y:.5f}%<extra></extra>')
    report.chart('recall-cost',fig,'Validation recall versus candidate cost (automatic, zoomed axes)',
                 'Candidate pairs','Blocking recall (%)',height=520)
    fig=go.Figure()
    for field in ['changed_name_fields','changed_address_fields']:
        fig.add_bar(x=metrics['variant'].to_list(),y=metrics[field].to_list(),name=field)
    report.chart('changed-fields',fig,'Validation field views changed relative to baseline',
                 'Variant','Changed field instances',barmode='group')
    baseline=metrics.filter(pl.col('variant')=='baseline').to_dicts()[0]
    for row in metrics.to_dicts():
        report.note(f"{row['variant']}: {row['pair_recall']:.5%} blocking recall; {row['candidate_pairs']:,} candidates; {row['missed_true_pairs']} missed true pairs. Candidate delta versus baseline: {row['candidate_pairs']-baseline['candidate_pairs']:+,}.")
    """)
    md("""
    ## 4. What was recovered, and what was newly lost?

    Compare the **same labeled pair** before and after cleaning. Net recall alone can hide a
    recovered match offset by a newly lost match. Display both counts and the affected records.
    A change can occur even when neither record was directly cleaned, because training IDF weights
    and competing target vectors also change.
    """)
    code("""
    report.table('paired_summary',metrics.select('variant','recovered_true_pairs','newly_lost_true_pairs','added_candidates','removed_candidates'))
    fig=go.Figure()
    for column,color in [('recovered_true_pairs','#059669'),('newly_lost_true_pairs','#dc2626')]:
        fig.add_bar(x=metrics['variant'].to_list(),y=metrics[column].to_list(),name=column,marker_color=color)
    report.chart('recovered-lost',fig,'True matches recovered and newly lost versus baseline',
                 'Variant','Labeled pairs',barmode='group')
    changed_true=pl.read_csv(RUN/'changed_true_pairs.csv',infer_schema=False)
    report.table('changed_true_pairs',changed_true,limit=15)
    churn_parts=[]
    for variant in variants:
        part=pl.read_parquet(RUN/'validation'/variant/'candidate_changes.parquet')
        if part.height:churn_parts.append(part)
    if churn_parts:
        churn=pl.concat(churn_parts).group_by('variant','change','is_true_match').len(name='pairs').sort('variant','change','is_true_match')
    else:
        churn=pl.DataFrame(schema={'variant':pl.String,'change':pl.String,'is_true_match':pl.Boolean,'pairs':pl.UInt32})
    report.table('candidate_churn',churn)
    fig=go.Figure()
    for change in ['added','removed']:
        for is_true in [True,False]:
            part=churn.filter((pl.col('change')==change)&(pl.col('is_true_match')==is_true))
            fig.add_bar(x=part['variant'].to_list(),y=part['pairs'].to_list(),name=change+(' true pairs' if is_true else ' nonmatches'))
    report.chart('candidate-churn',fig,'How the full candidate pool changes',
                 'Variant','Added or removed candidate pairs',barmode='group')
    for row in metrics.to_dicts():
        if row['variant']!='baseline':
            report.note(f"{row['variant']} adds {row['added_candidates']:,} candidates and removes {row['removed_candidates']:,}; a small net count change does not mean the candidate identities stayed the same.")
    """)
    md("""
    ## 5. Is the effect consistent across countries and target sources?

    Slices retain their actual denominators. These are descriptive comparisons, not independent
    statistical tests or proof of generalization to unseen countries. The evaluation split remains reserved.
    """)
    code("""
    slices=pl.read_csv(RUN/'validation_slices.csv')
    report.table('coverage_slices',slices.select('method','slice_type','slice','true_pairs','missed_true_pairs','pair_recall'),limit=15)
    fig=go.Figure()
    for variant in variants:
        part=slices.filter(pl.col('method')==variant)
        fig.add_bar(x=[row['slice_type']+': '+row['slice'] for row in part.to_dicts()],
                    y=part['missed_true_pairs'].to_list(),name=variant,marker_color=colors[variant])
    report.chart('slice-misses',fig,'Missed true pairs by country and target source',
                 'Slice','Missed labeled pairs',barmode='group')
    """)
    md("""
    ## 6. What does a candidate list look like before and after cleaning?

    Pick a reference business. The chart shows the best rank across the name/address views in its
    target source. Rank 1 is closest; **11 means not retrieved**, not a score or rank from search.
    Hover for the target text, truth label, view-specific ranks, and maximum cosine.

    Examples prioritize recovered/newly lost pairs, then large candidate-list changes. They are
    diagnostic selections, not prevalence estimates. Show eight top-ranked candidates per variant
    plus all true matches. Full candidate sets remain in the TSV/Parquet outputs.
    """)
    code("""
    sources={name:pl.read_csv(DATA/'validation'/f'{name}.tsv',separator='\\t',infer_schema=False)
             for name in ['source1','source2','source3']}
    refs={row['entity_id']:row for row in sources['source1'].to_dicts()}
    targets={row['entity_id']:row for source in ['source2','source3'] for row in sources[source].to_dicts()}
    truth={key:set(value.split(',')) if value else set()
           for key,value in pl.read_csv(DATA/'validation/ground_truth.tsv',separator='\\t',infer_schema=False).iter_rows()}
    choices=[]
    for change in ['recovered','newly_lost']:
        for key in changed_true.filter(pl.col('change')==change)['source1_entity_id'].unique(maintain_order=True).head(2):
            if key not in choices:choices.append(key)
    for variant in variants[1:]:
        paired=pl.read_csv(RUN/'validation'/variant/'paired_changes.csv').with_columns(
            (pl.col('added_candidates')+pl.col('removed_candidates')).alias('churn'))
        for key in paired.sort(['churn','source1_entity_id'],descending=[True,False])['source1_entity_id'].head(3):
            if key not in choices:choices.append(key)
    choices=choices[:6]
    report.table('explorer_references',sources['source1'].filter(pl.col('entity_id').is_in(choices)))
    evidence={variant:pl.read_parquet(RUN/'validation'/variant/'retrieval_evidence.parquet') for variant in variants}
    fig=go.Figure();buttons=[]
    for query_index,key in enumerate(choices):
        grouped={};shown=set(truth[key])
        for variant in variants:
            items={}
            for row in evidence[variant].filter(pl.col('source1_entity_id')==key).to_dicts():
                item=items.setdefault(row['target_id'],{'best_rank':11,'cosine':0.,'name_rank':'not retrieved','address_rank':'not retrieved'})
                item['best_rank']=min(item['best_rank'],row['rank']);item['cosine']=max(item['cosine'],row['cosine'])
                item[row['view']+'_rank']=str(row['rank'])
            grouped[variant]=items
            shown.update(sorted(items,key=lambda target:(items[target]['best_rank'],-items[target]['cosine'],target))[:8])
        target_ids=sorted(shown)
        for variant in variants:
            items=grouped[variant]
            custom=[];ranks=[]
            for target in target_ids:
                item=items.get(target,{'best_rank':11,'cosine':None,'name_rank':'not retrieved','address_rank':'not retrieved'})
                ranks.append(item['best_rank'])
                custom.append([escape(targets[target]['business_name'] or ''),escape(targets[target]['business_address'] or ''),
                               str(target in truth[key]),item['name_rank'],item['address_rank'],
                               f"{item['cosine']:.4f}" if item['cosine'] is not None else 'not retrieved'])
            fig.add_scatter(x=ranks,y=target_ids,mode='markers',name=variant,visible=query_index==0,
                marker=dict(size=11,color=colors[variant],symbol=['circle','diamond','square'][variants.index(variant)]),
                customdata=custom,hovertemplate='%{y}<br>%{customdata[0]}<br>%{customdata[1]}<br>True match: %{customdata[2]}<br>Name rank: %{customdata[3]} | Address rank: %{customdata[4]}<br>Maximum cosine: %{customdata[5]}<extra>%{fullData.name}</extra>')
        buttons.append(dict(label=key+' / '+refs[key]['country'],method='update',args=[
            {'visible':[i//len(variants)==query_index for i in range(len(choices)*len(variants))]},
            {'title':'Before/after candidate ranks for '+key}]))
    fig.update_xaxes(tickvals=[1,2,4,6,8,10,11],ticktext=['1','2','4','6','8','10','Not retrieved'])
    fig.add_vline(x=10.5,line_dash='dot')
    report.chart('candidate-before-after',fig,'Before/after candidate ranks for '+choices[0],
                 'Best rank across retrieval views (11 = absent)','Target record ID',height=760,
                 updatemenus=[dict(buttons=buttons,x=1,y=1.14,xanchor='right')])
    """)
    md("""
    ## 7. Runtime and decision

    Timings are single-run observations with a common configuration. Fit, validation preparation,
    and retrieval are reported separately; export and training-candidate generation are excluded
    from this chart. Baseline training candidates were copied from the verified saved baseline;
    new variants have newly generated training candidates. Do not compare those training export
    times as an algorithm benchmark. Peak memory is not measured.

    The original baseline stays adopted. A suggested cleaning variant can be carried into a later
    model comparison, but this blocking experiment does not establish its final precision/F0.5.
    """)
    code("""
    from run_cleaning import read_candidates
    training_truth={key:set(value.split(',')) if value else set()
        for key,value in pl.read_csv(DATA/'train/ground_truth.tsv',separator='\\t',infer_schema=False).iter_rows()}
    before=read_candidates(RUN/'train/baseline/candidate_pairs.tsv')
    training_changes=[]
    for variant in variants:
        after=read_candidates(RUN/'train'/variant/'candidate_pairs.tsv')
        training_changes.append({'variant':variant,
            'recovered_true_pairs':sum(len((after[key]-before[key])&training_truth[key]) for key in training_truth),
            'newly_lost_true_pairs':sum(len((before[key]-after[key])&training_truth[key]) for key in training_truth)})
    training_paired=pl.DataFrame(training_changes)
    report.table('training_paired_changes',training_paired)
    for row in training_changes:
        if row['variant']!='baseline':
            report.note(f"In-sample training check for {row['variant']}: {row['recovered_true_pairs']} recovered true pairs and {row['newly_lost_true_pairs']} newly lost true pairs. These are not held-out performance estimates.")
    report.table('runtime',metrics.select('variant','fit_seconds','validation_prepare_seconds','retrieval_seconds'))
    fig=go.Figure()
    for field in ['fit_seconds','validation_prepare_seconds','retrieval_seconds']:
        fig.add_bar(x=metrics['variant'].to_list(),y=metrics[field].to_list(),name=field)
    report.chart('runtime',fig,'Measured setup and validation retrieval time','Variant','Seconds',barmode='stack')
    for row in metrics.to_dicts():
        if row['variant']!='baseline':
            report.note(f"{row['variant']}: recovered {row['recovered_true_pairs']} baseline-missed pairs and newly lost {row['newly_lost_true_pairs']} baseline-retained pairs. Training recall is {row['training_recall']:.3%}, an in-sample diagnostic.")
    suggestion=run['suggested_cleaning_variant_for_model_comparison']
    report.note('The original baseline remains adopted. Suggested cleaned variant for later model comparison: '+str(suggestion)+'. The suggestion prioritizes no newly lost baseline pairs, then fewer misses, then candidate cost; it is not a final-model selection.')
    report.note('The minimum-count suggestion differs by only one candidate between the two cleaned variants and is a mechanical tie-break, not meaningful evidence that abbreviation expansion is better. Neither variant improves validation recall; both also lose some training true pairs. Preserve the baseline and keep cleaning variants as controlled model-comparison branches.')
    report.note('The audit/rules were frozen before validation access. No broad transliteration, typo correction, classifier tuning, evaluation access, or raw-data modification was performed.')
    findings='# Cleaning and blocking findings\\n\\n'+'\\n'.join('- '+item for item in report.findings)
    findings+='\\n\\nLimits: results concern the reduced validation pool and a fixed K=10 configuration. Refitting training-only TF-IDF is part of each variant. Candidate changes may affect a future classifier differently; final match precision/recall/F0.5 have not been measured.\\n'
    (report.folder/'findings.md').write_text(findings,encoding='utf-8')
    (report.folder/'report_manifest.json').write_text(json.dumps({'run_dir':str(RUN),'charts':report.figures,
        'tables':report.table_names,'scope':run['scope'],'classifier_trained':False,'evaluation_opened':False},indent=2)+'\\n',encoding='utf-8')
    display(Markdown('All three variants and their training/validation candidates are preserved for later controlled model comparisons.'))
    """)
    return nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3 (ML lab uv environment)','language':'python','name':'python3'},
        'language_info':{'name':'python','version':sys.version.split()[0]}})


def main(run_dir):
    notebook=build_notebook(Path(run_dir).resolve())
    path=PROJECT/'notebooks/03_cleaning_ablation.ipynb'
    folder=PROJECT/'reports/cleaning';folder.mkdir(parents=True,exist_ok=True)
    os.environ['JUPYTER_RUNTIME_DIR']=str(PROJECT/'.jupyter_runtime')
    nbformat.write(notebook,path)
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    try:
        NotebookClient(notebook,km=manager,timeout=180,startup_timeout=60,
                       resources={'metadata':{'path':str(PROJECT)}}).execute()
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    nbformat.validate(notebook);nbformat.write(notebook,path)
    exporter=HTMLExporter();exporter.exclude_input=True
    body,_=exporter.from_notebook_node(notebook)
    document=BeautifulSoup(body,'html.parser')
    for script in document.find_all('script',src=True):script.decompose()
    (folder/'cleaning_ablation.html').write_text(str(document),encoding='utf-8')
    print(f'Executed notebook: {path}\nOffline report: {folder / "cleaning_ablation.html"}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'artifacts/cleaning_v1')
    main(parser.parse_args().run_dir)
