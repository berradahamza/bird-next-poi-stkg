"""Paired visitor-level bootstrap comparison of two frozen run reports."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run_a',type=Path);p.add_argument('run_b',type=Path)
    p.add_argument('--fold',choices=['validation','test'],default='validation')
    p.add_argument('--metric',choices=['mrr','acc1','hit5','hit10'],default='mrr')
    p.add_argument('--resamples',type=int,default=10000);p.add_argument('--seed',type=int,default=2022)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    a=json.loads((args.run_a/'config.json').read_text());b=json.loads((args.run_b/'config.json').read_text())
    for key in ('split_seed','processed_input_sha256','availability_mask','max_history'):
        if a[key]!=b[key]:raise ValueError(f"Incomparable runs, mismatch in {key}")
    def report(root):
        path=root/('metrics.json' if args.fold=='validation' else 'test_metrics.json')
        return json.loads(path.read_text())['validation' if args.fold=='validation' else 'test']
    x,y=report(args.run_a)['per_visitor'],report(args.run_b)['per_visitor']
    if set(x)!=set(y):raise ValueError('Different visitors in run reports')
    ids=sorted(x);diff=np.array([x[v][args.metric]-y[v][args.metric] for v in ids])
    rng=np.random.default_rng(args.seed)
    values=np.mean(rng.choice(diff,(args.resamples,len(ids)),replace=True),axis=1)
    out={'fold':args.fold,'metric':args.metric,'run_a':str(args.run_a),'run_b':str(args.run_b),
         'visitor_count':len(ids),'macro_mean_difference_a_minus_b':float(diff.mean()),
         'paired_visitor_bootstrap_95pct':[float(np.quantile(values,.025)),float(np.quantile(values,.975))],
         'resamples':args.resamples,'interpretation':'Exploratory if method was chosen using these visitors; CI not a substitute for independent validation'}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
