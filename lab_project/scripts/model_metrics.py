"""Exact validation threshold selection and end-to-end set evaluation."""

import numpy as np
import polars as pl

from er_metrics import f05,set_metrics


def predictions_at(frame,scores,truth,threshold):
    result={key:set() for key in truth}
    for key,target,score in zip(frame['source1_entity_id'],frame['target_id'],scores):
        if score>=threshold:result[key].add(target)
    return result


def threshold_curve(frame,scores,truth):
    """Sweep distinct scores, processing tied scores together. Blocked truths stay FN."""
    scores=np.asarray(scores,dtype=float)
    if len(scores)!=frame.height or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected one finite score per candidate')
    keys=sorted(truth);lookup={key:i for i,key in enumerate(keys)}
    qi=np.array([lookup[key] for key in frame['source1_entity_id']],dtype=int)
    y=np.array([target in truth[key] for key,target in frame.select('source1_entity_id','target_id').iter_rows()],dtype=int)
    total_true=sum(map(len,truth.values()));n=len(keys)
    true_counts=[len(truth[key]) for key in keys];tp=[0]*n;counts=[0]*n
    per_score=[1. if not count else 0. for count in true_counts]
    per_exact=[not count for count in true_counts]
    score_sum=sum(per_score);exact=sum(per_exact)
    singleton_count=sum(not count for count in true_counts);singleton_correct=singleton_count
    total_tp=total_fp=0;rows=[]

    def record(threshold):
        rows.append({'threshold':float(threshold),'macro_f05':float(np.clip(score_sum/n,0,1)),
            'micro_precision':total_tp/(total_tp+total_fp) if total_tp+total_fp else 0.,
            'micro_recall':total_tp/total_true if total_true else None,
            'micro_f1':2*total_tp/(total_true+total_tp+total_fp) if total_true+total_tp+total_fp else 0.,
            'exact_set_accuracy':exact/n,'singleton_correctness':singleton_correct/singleton_count if singleton_count else None,
            'tp':total_tp,'fp':total_fp,'fn':total_true-total_tp})
    record(np.nextafter(float(scores.max()),np.inf))
    order=np.argsort(-scores,kind='stable')
    for position,index in enumerate(order):
        q=int(qi[index]);old_count=counts[q]
        score_sum-=per_score[q];exact-=int(per_exact[q])
        if true_counts[q]==0 and old_count==0:singleton_correct-=1
        counts[q]+=1;tp[q]+=int(y[index]);total_tp+=int(y[index]);total_fp+=1-int(y[index])
        per_score[q]=1.25*tp[q]/(counts[q]+.25*true_counts[q])
        per_exact[q]=counts[q]==true_counts[q] and tp[q]==true_counts[q]
        score_sum+=per_score[q];exact+=int(per_exact[q])
        if position==len(order)-1 or scores[order[position+1]]!=scores[index]:record(scores[index])
    curve=pl.DataFrame(rows)
    best=min(rows,key=lambda row:(-round(row['macro_f05'],12),-row['micro_precision'],-row['threshold']))
    return curve,best


def grid_curve(curve,chosen_threshold):
    """Exact metrics at 201 display thresholds plus the selected exact threshold."""
    rows=sorted(curve.to_dicts(),key=lambda row:-row['threshold'])
    thresholds=sorted(set([float(value) for value in np.linspace(0,1,201)]+[float(chosen_threshold)]))
    result=[]
    # Curve states become active when their score is >= the requested threshold.
    for threshold in thresholds:
        eligible=[row for row in rows[1:] if row['threshold']>=threshold]
        state=eligible[-1] if eligible else rows[0]
        result.append({**state,'threshold':threshold})
    return result


def score_summary(truth,predictions):
    result=set_metrics(truth,predictions)
    result['singleton_false_merge_rate']=1-result['singleton_correctness'] if result['singleton_correctness'] is not None else None
    result['mean_predicted_matches']=sum(map(len,predictions.values()))/len(truth)
    return result


def bootstrap_macro(truth,predictions,seed=42,repeats=2000):
    values=np.array([f05(truth[key],predictions[key]) for key in sorted(truth)])
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(repeats)])
    return {'low':float(np.quantile(means,.025)),'high':float(np.quantile(means,.975)),
            'seed':seed,'repeats':repeats,'unit':'Source 1 business','scope':'fixed selected pipeline; excludes training/selection uncertainty'}
