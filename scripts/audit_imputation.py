"""Pseudo-mask measured ends to evaluate the requested imputation empirically."""
from __future__ import annotations
import json,math,statistics
from pathlib import Path
import numpy as np
from stkg_cll_poi.preprocess import put_json

ROOT=Path(__file__).resolve().parent.parent

def describe(errors):
    x=np.asarray(errors,dtype=float)
    return {'n':len(x),'mae_seconds':float(np.abs(x).mean()),
            'median_absolute_error_seconds':float(np.median(np.abs(x))),
            'rmse_seconds':float(np.sqrt(np.square(x).mean())),
            'p90_absolute_error_seconds':float(np.quantile(np.abs(x),.9))}

def main():
    d=json.loads((ROOT/'data/processed/trajectories_poi.json').read_text())
    retrospective=[];prefix=[];gap_only=[];coverage=0;cap_offline=0;cap_prefix=0
    for v in d['visitors'].values():
        events=v['events']
        for i,e in enumerate(events[:-1]):
            true=e['duration_observed_seconds']
            if true is None or true<=0:continue
            gap=events[i+1]['t_start_seconds']-e['t_start_seconds']
            other=[x['duration_observed_seconds'] for j,x in enumerate(events) if j!=i and x['duration_observed_seconds'] is not None and x['duration_observed_seconds']>0]
            if not other:continue
            full_est=min(gap,(gap+statistics.mean(other))/2)
            retrospective.append(full_est-true)
            cap_offline+=full_est>=gap-1e-8
            gap_only.append(gap-true)
            seen=[x['duration_observed_seconds'] for x in events[:i] if x['duration_observed_seconds'] is not None and x['duration_observed_seconds']>0 and x['t_end_observed_seconds']<=events[i+1]['t_start_seconds']+1e-9]
            if seen:
                online_est=min(gap,(gap+statistics.mean(seen))/2)
                prefix.append(online_est-true)
                cap_prefix+=online_est>=gap-1e-8
            coverage+=bool(seen)
    result={'method':'Pseudo-mask one observed end at a time. Each visit full mean excludes the heldout measurement; prefix mean excludes current/future ends. This evaluates observed ends, which may differ systematically from truly missing ends.',
            'retrospective_full_visit':describe(retrospective),
            'causal_prefix_available_subset':describe(prefix),
            'causal_prefix_coverage_among_retrospective_cases':coverage/len(retrospective),
            'next_start_gap_baseline':describe(gap_only),
            'retrospective_at_gap_cap':cap_offline,
            'causal_at_gap_cap':cap_prefix}
    put_json(ROOT/'runs/imputation_sensitivity.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
