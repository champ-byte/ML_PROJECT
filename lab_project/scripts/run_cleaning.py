"""Training-only cleaning audit and frozen-K blocking ablation; no classifier work."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time

import polars as pl

from blocking import TfidfBlocker, candidate_map
from cleaning import VARIANTS, audit, clean_for_retrieval, normalizer, rules
from er_metrics import blocking_metrics
from run_blocking import DATA, PROJECT, DEFAULT_OUTPUT as BASELINE, details, diagnose_misses, load_split, slice_metrics, write_candidates


OUTPUT = PROJECT / 'artifacts/cleaning_v1'


def read_candidates(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    result = {row['source1_entity_id']: set(filter(None,row['candidate_entity_ids'].split(','))) for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate reference rows in candidate file')
    return result


def paired_changes(truth, before, after, sources, variant):
    queries = {row['entity_id']:row for row in sources['source1'].to_dicts()}
    targets = {row['entity_id']:row for source in ('source2','source3') for row in sources[source].to_dicts()}
    per_query, changed_pairs, true_changes = [], [], []
    for key in sorted(truth):
        added, removed = after[key] - before[key], before[key] - after[key]
        recovered, lost = added & truth[key], removed & truth[key]
        per_query.append({'variant':variant,'source1_entity_id':key,'country':queries[key]['country'],
            'before_candidates':len(before[key]),'after_candidates':len(after[key]),
            'added_candidates':len(added),'removed_candidates':len(removed),
            'recovered_true_pairs':len(recovered),'newly_lost_true_pairs':len(lost)})
        for change, values in [('added',added),('removed',removed)]:
            changed_pairs.extend({'variant':variant,'source1_entity_id':key,'target_id':target,
                                  'change':change,'is_true_match':target in truth[key]} for target in sorted(values))
        for change, values in [('recovered',recovered),('newly_lost',lost)]:
            for target in sorted(values):
                row = {'variant':variant,'source1_entity_id':key,'target_id':target,'change':change,
                       'country':queries[key]['country'],'target_source': 'source2' if target.startswith('S2-') else 'source3'}
                for label, record in [('reference',queries[key]),('target',targets[target])]:
                    for field in ('business_name','business_address'):
                        row[f'{label}_{field}_raw'] = record[field]
                        row[f'{label}_{field}_baseline'] = clean_for_retrieval(record[field],field,'baseline')
                        row[f'{label}_{field}_cleaned'] = clean_for_retrieval(record[field],field,variant)
                true_changes.append(row)
    churn_schema={'variant':pl.String,'source1_entity_id':pl.String,'target_id':pl.String,'change':pl.String,'is_true_match':pl.Boolean}
    true_schema={'variant':pl.String,'source1_entity_id':pl.String,'target_id':pl.String,'change':pl.String,'country':pl.String,'target_source':pl.String,
                 **{f'{label}_{field}_{view}':pl.String for label in ('reference','target') for field in ('business_name','business_address') for view in ('raw','baseline','cleaned')}}
    return pl.DataFrame(per_query), pl.DataFrame(changed_pairs,schema=churn_schema), pl.DataFrame(true_changes,schema=true_schema)


def run(output_dir=OUTPUT):
    output_dir=Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f'Output exists; choose a new --output-dir: {output_dir}')
    output_dir.mkdir(parents=True)
    marker=output_dir/'INCOMPLETE.txt'
    marker.write_text('Cleaning experiment incomplete.\n',encoding='utf-8')
    started=time.perf_counter()
    try:
        baseline_meta=json.loads((BASELINE/'run.json').read_text(encoding='utf-8'))
        if baseline_meta['selected_method']!='tfidf_union_k10':
            raise ValueError('This frozen experiment expects the saved K=10 baseline')
        for relative, expected in baseline_meta['artifact_sha256'].items():
            if hashlib.sha256((BASELINE/relative).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Saved baseline artifact changed: {relative}')
        manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
        training,training_truth,train_hashes=load_split(DATA,'train',manifest)
        # Audit and freeze all rule definitions BEFORE opening validation records.
        audit_folder=output_dir/'training_audit';audit_folder.mkdir()
        for name,table in audit(training).items(): table.write_csv(audit_folder/f'{name}.csv')
        rule_path=output_dir/'rules.json'
        rule_path.write_text(json.dumps(rules(),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        rule_hash=hashlib.sha256(rule_path.read_bytes()).hexdigest()
        print('Training audit saved; cleaning rules frozen before validation load.',flush=True)
        validation,truth,val_hashes=load_split(DATA,'validation',manifest)
        if {**train_hashes,**val_hashes} != baseline_meta['input_hashes']:
            raise ValueError('Input files differ from saved baseline')
        saved=read_candidates(BASELINE/'validation/tfidf_union_k10/candidate_pairs.tsv')
        valid_targets=set(validation['source2']['entity_id'])|set(validation['source3']['entity_id'])
        train_targets=set(training['source2']['entity_id'])|set(training['source3']['entity_id'])
        summaries,view_rows,slices,changes=[],[],[],[]
        source_code={name:hashlib.sha256((PROJECT/'scripts'/name).read_bytes()).hexdigest()
                     for name in ('cleaning.py','blocking.py','run_cleaning.py','er_metrics.py')}
        for variant in VARIANTS:
            fit_start=time.perf_counter()
            blocker=TfidfBlocker(batch_size=64,normalizer=normalizer(variant)).fit(training)
            fit_seconds=time.perf_counter()-fit_start
            prep_start=time.perf_counter();prepared=blocker.prepare(validation)
            prep_seconds=time.perf_counter()-prep_start
            retrieval_start=time.perf_counter();evidence,diagnostics=blocker.retrieve(prepared,10)
            retrieval_seconds=time.perf_counter()-retrieval_start
            candidates=candidate_map(sorted(truth),evidence)
            if variant=='baseline' and candidates!=saved:
                raise ValueError('Baseline reproduction differs from saved experiment; stop before comparisons')
            folder=output_dir/'validation'/variant
            write_candidates(folder,candidates,valid_targets)
            evidence.write_parquet(folder/'retrieval_evidence.parquet')
            diagnostics.write_csv(folder/'query_diagnostics.csv')
            details(truth,candidates,validation['source1']).write_csv(folder/'per_business.csv')
            diagnose_misses(truth,candidates,prepared).write_csv(folder/'missed_true_pairs.csv')
            metrics=blocking_metrics(truth,candidates,len(valid_targets))
            per_query,churn,true_changes=paired_changes(truth,saved,candidates,validation,variant)
            per_query.write_csv(folder/'paired_changes.csv')
            churn.write_parquet(folder/'candidate_changes.parquet')
            true_changes.write_csv(folder/'changed_true_pairs.csv')
            recovered=int(per_query['recovered_true_pairs'].sum());lost=int(per_query['newly_lost_true_pairs'].sum())
            changed_fields={field:sum(clean_for_retrieval(value,field,variant)!=clean_for_retrieval(value,field,'baseline')
                                     for frame in validation.values() for value in frame[field])
                            for field in ('business_name','business_address')}
            row={'variant':variant,**metrics,'recovered_true_pairs':recovered,'newly_lost_true_pairs':lost,
                 'added_candidates':int(per_query['added_candidates'].sum()),
                 'removed_candidates':int(per_query['removed_candidates'].sum()),
                 'changed_name_fields':changed_fields['business_name'],
                 'changed_address_fields':changed_fields['business_address'],
                 'fit_seconds':fit_seconds,'validation_prepare_seconds':prep_seconds,
                 'retrieval_seconds':retrieval_seconds}
            print(f"{variant}: recall={metrics['pair_recall']:.5%}, candidates={metrics['candidate_pairs']:,}, recovered={recovered}, newly_lost={lost}",flush=True)
            for view in ('name','address','union'):
                view_rows.append({'variant':variant,'view':view,**blocking_metrics(truth,candidate_map(sorted(truth),evidence,None if view=='union' else view),len(valid_targets))})
            slices.extend(slice_metrics(truth,candidates,validation,variant))
            changes.append(true_changes)
            with (folder/'tfidf_vectorizers.pkl').open('wb') as stream:pickle.dump(blocker.vectorizers,stream)
            print(f'{variant}: preparing training candidates for later model comparisons...',flush=True)
            train_folder=output_dir/'train'/variant;train_folder.mkdir(parents=True)
            if variant=='baseline':
                for filename in ('candidate_pairs.tsv','retrieval_evidence.parquet','query_diagnostics.csv'):
                    shutil.copy2(BASELINE/'train/tfidf_union_k10'/filename,train_folder/filename)
                training_candidates=read_candidates(train_folder/'candidate_pairs.tsv')
                row['training_candidate_origin']='copied from verified saved baseline'
            else:
                train_prepared=blocker.prepare(training)
                train_evidence,train_diagnostics=blocker.retrieve(train_prepared,10)
                training_candidates=candidate_map(sorted(training_truth),train_evidence)
                write_candidates(train_folder,training_candidates,train_targets)
                train_evidence.write_parquet(train_folder/'retrieval_evidence.parquet')
                train_diagnostics.write_csv(train_folder/'query_diagnostics.csv')
                row['training_candidate_origin']='generated with training-fitted variant'
            training_metrics=blocking_metrics(training_truth,training_candidates,len(train_targets))
            row['training_recall']=training_metrics['pair_recall']
            row['training_candidate_pairs']=training_metrics['candidate_pairs']
            summaries.append(row)
        metrics=pl.DataFrame(summaries)
        metrics.write_csv(output_dir/'validation_comparison.csv')
        pl.DataFrame(view_rows).write_csv(output_dir/'validation_view_metrics.csv')
        pl.DataFrame(slices).write_csv(output_dir/'validation_slices.csv')
        pl.concat(changes).write_csv(output_dir/'changed_true_pairs.csv')
        # Comparison selection is provisional: no cleaning variant is adopted for final matching yet.
        nonregressive=[row for row in summaries if row['variant']!='baseline' and row['newly_lost_true_pairs']==0]
        suggested=min(nonregressive,key=lambda row:(row['missed_true_pairs'],row['candidate_pairs'],row['variant'])) if nonregressive else None
        for relative,expected in {**train_hashes,**val_hashes}.items():
            if hashlib.sha256((DATA/relative).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Input changed during experiment: {relative}')
        result={'status':'complete','scope':'training audit and TF-IDF fit; validation comparison only',
            'protocol':{'k':10,'ngram_range':[3,5],'batch_size':64,'analyzer':'char','dtype':'float32',
                        'variants':list(VARIANTS),'refit_on_training_per_variant':True,
                        'rules_frozen_before_validation_load':True,'rules_sha256':rule_hash,
                        'raw_records_modified':False,'country_partition_modified':False,
                        'no_truth_injection':True,'no_validation_label_driven_rule_tuning':True},
            'versions':baseline_meta['versions'],'input_hashes':{**train_hashes,**val_hashes},
            'baseline_run':str(BASELINE),'baseline_reproduced_exactly':True,'script_hashes':source_code,
            'adopted_variant':'baseline','automatic_adoption':False,
            'suggested_cleaning_variant_for_model_comparison':suggested['variant'] if suggested else None,
            'suggestion_rule':'Among variants with no newly lost baseline true pairs, minimize missed pairs, then candidate count. This is not a final-model selection.',
            'classifier_trained':False,'evaluation_opened':False,'elapsed_seconds':time.perf_counter()-started,
            'memory_note':'Peak memory not measured; variants run sequentially; sparse products use batches of 64 references.',
            'artifact_sha256':{path.relative_to(output_dir).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(output_dir.rglob('*')) if path.is_file() and path.name!='INCOMPLETE.txt'}}
        (output_dir/'run.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        marker.unlink()
        print(f"Complete in {result['elapsed_seconds']:.1f}s; original baseline remains adopted.",flush=True)
        return result
    except Exception as error:
        marker.write_text(f'Experiment failed: {error}\n',encoding='utf-8')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    run(parser.parse_args().output_dir)
