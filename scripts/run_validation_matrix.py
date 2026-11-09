"""Predeclared, visitor-fixed comparison. Does not open the neural test set.

Usage: python scripts/run_validation_matrix.py --seeds 2022 2023 2024
Run only after deciding configs; changing them based on test results invalidates a
confirmatory interpretation of the test set.
"""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
EXPERIMENTS=[
    ('poi_id_gru', ['--variant','poi_id','--seq-weight','0','--graph-weight','0','--no-augmentation']),
    ('markov_gru_transitions', ['--variant','transitions','--seq-weight','0','--graph-weight','0','--no-augmentation']),
    ('full_supervised', ['--variant','full','--seq-weight','0','--graph-weight','0','--no-augmentation']),
    ('full_aug_only', ['--variant','full','--seq-weight','0','--graph-weight','0']),
    ('full_seq_cl', ['--variant','full','--graph-weight','0']),
    ('full_graph_cl', ['--variant','full','--seq-weight','0']),
    ('full_joint_cl', ['--variant','full']),
    ('no_subject', ['--variant','no_subject']),
    ('main_only', ['--variant','main_only']),
    ('no_time', ['--variant','no_time']),
    ('no_behavior', ['--variant','no_behavior']),
    ('no_room', ['--variant','no_room']),
    ('shuffled_subjects', ['--variant','shuffled_subjects']),
    ('sasrec_style', ['--architecture','sasrec_style','--variant','poi_id','--seq-weight','0','--graph-weight','0','--no-augmentation']),
    ('temporal_transformer', ['--architecture','temporal_transformer','--variant','poi_id','--seq-weight','0','--graph-weight','0','--no-augmentation']),
]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=[2022,2023,2024])
    parser.add_argument('--split-seed',type=int,default=2022)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--output-root',type=Path,default=ROOT/'runs/validation_matrix')
    parser.add_argument('--only',nargs='*',choices=[x[0] for x in EXPERIMENTS])
    args=parser.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=True)
    planned=[(name,extra) for name,extra in EXPERIMENTS if args.only is None or name in args.only]
    rows=[]
    for name,extra in planned:
        for seed in args.seeds:
            run=args.output_root/f'{name}_split{args.split_seed}_init{seed}'
            command=[sys.executable,'-m','stkg_cll_poi','train','--seed',str(seed),
                     '--split-seed',str(args.split_seed),'--epochs',str(args.epochs),
                     '--output',str(run),'--bootstrap','0',*extra]
            print('RUN',name,seed,flush=True)
            subprocess.run(command,cwd=ROOT,check=True)
            metrics=json.loads((run/'metrics.json').read_text())
            rows.append({'method':name,'split_seed':args.split_seed,'init_seed':seed,
                         'epoch':metrics['selected_epoch'],'validation_mrr':metrics['validation']['micro']['mrr'],
                         'validation_acc1':metrics['validation']['micro']['acc1']})
            with (args.output_root/'validation_matrix.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Validation-only matrix written:',args.output_root/'validation_matrix.csv')

if __name__=='__main__':main()
