"""python -m stkg_cll_poi {prepare,baselines,train,predict}."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .experiment import baselines,evaluate_test,inference,train
from .features import save_split_and_graph
from .preprocess import prepare
from .preprocess import put_json
from .heterogeneous import HETERO_VARIANTS, build_heterogeneous_graph, graph_summary, print_graph_summary
from .spatial import DEFAULT_ADJACENCY, load_spatial
from .experiment import load_processed

ROOT=Path(__file__).resolve().parent.parent

def main():
    parser=argparse.ArgumentParser(description="Auditable BIRD STKG/contrastive next-POI experiment")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare",help="Build canonical JSON, pairing report, visitor split and train graph")
    p.add_argument("--bird-root",type=Path,default=ROOT/"data/raw/BIRD")
    p.add_argument("--semantic-csv",type=Path,default=ROOT/"data/raw/oeuvres_semantique.csv")
    p.add_argument("--data-dir",type=Path,default=ROOT/"data/processed")
    p.add_argument("--seed",type=int,default=2022)
    p.add_argument("--spatial-json",type=Path,default=DEFAULT_ADJACENCY)
    p.add_argument("--strict-spatial",action="store_true")
    spatial_parser=sub.add_parser("validate-spatial",help="Validate neighbors, adjacency attributes and floor geometry")
    spatial_parser.add_argument("--spatial-json",type=Path,default=DEFAULT_ADJACENCY)
    spatial_parser.add_argument("--strict",action="store_true")
    spatial_parser.add_argument("--output",type=Path,default=ROOT/"data/processed/spatial_validation.json")
    graph_parser=sub.add_parser("build-graph",help="Build and inspect a typed graph without training")
    graph_parser.add_argument("--data-dir",type=Path,default=ROOT/"data/processed")
    graph_parser.add_argument("--output",type=Path,default=ROOT/"data/processed/heterogeneous")
    graph_parser.add_argument("--variant",choices=HETERO_VARIANTS,default="poi_room_subject")
    graph_parser.add_argument("--split-seed",type=int,default=2022)
    b=sub.add_parser("baselines",help="Full-ranking popularity, Markov-1 and Markov-2")
    b.add_argument("--data-dir",type=Path,default=ROOT/"data/processed")
    b.add_argument("--output",type=Path,default=ROOT/"runs/baselines_seed2022")
    b.add_argument("--seed",type=int,default=2022)
    b.add_argument("--history",type=int,default=32)
    b.add_argument("--bootstrap",type=int,default=1000)
    b.add_argument("--availability-mask",action="store_true")
    t=sub.add_parser("train",help="Train one predeclared variant, early-stop on visitor-disjoint validation")
    t.add_argument("--data-dir",type=Path,default=ROOT/"data/processed")
    t.add_argument("--output",type=Path,default=ROOT/"runs/hetero_seed2022")
    t.add_argument("--variant",choices=[*HETERO_VARIANTS,"full","poi_id","transitions","no_subject","main_only","no_time","no_behavior","no_room","shuffled_subjects"],default="poi_room_subject")
    for graph_options in (t,graph_parser):
        graph_options.add_argument("--spatial-json",type=Path,default=DEFAULT_ADJACENCY)
        graph_options.add_argument("--strict-spatial",action="store_true")
        graph_options.add_argument("--gnn-layers",type=int,choices=[1,2],default=2)
        graph_options.add_argument("--room-features",choices=["identity","metadata"],default="metadata")
        graph_options.add_argument("--midpoint",action="store_true")
    t.add_argument("--architecture",choices=["gru","sasrec_style","temporal_transformer"],default="gru")
    for flag,typ,default in [("seed",int,2022),("split-seed",int,2022),("hidden",int,64),("epochs",int,30),("patience",int,5),("batch-size",int,64),("history",int,32),
                             ("lr",float,.001),("weight-decay",float,.00005),("dropout",float,.2),("edge-dropout",float,.1),
                             ("seq-weight",float,.05),("graph-weight",float,.02),("temperature",float,.2),("bootstrap",int,1000)]:
        t.add_argument("--"+flag,type=typ,default=default)
    t.add_argument("--no-augmentation",action="store_true",help="One supervised pass and no graph edge perturbation; ordinary dropout remains. Requires both CL weights = 0")
    t.add_argument("--availability-mask",action="store_true")
    et=sub.add_parser("evaluate-test",help="Unseal one frozen run's test results exactly once")
    et.add_argument("--data-dir",type=Path,default=ROOT/"data/processed")
    et.add_argument("--run-dir",type=Path,required=True)
    et.add_argument("--bootstrap",type=int,default=None)
    q=sub.add_parser("predict",help="Predict from one prefix in an existing processed visit")
    q.add_argument("--data-dir",type=Path,default=ROOT/"data/processed")
    q.add_argument("--run-dir",type=Path,required=True)
    q.add_argument("--visit-id",required=True)
    q.add_argument("--event-index",required=True,type=int)
    q.add_argument("--topk",type=int,default=5)
    a=parser.parse_args()
    if a.command=="prepare":
        audit=prepare(a.bird_root,a.semantic_csv,a.data_dir)
        split,graph=save_split_and_graph(a.data_dir,a.seed)
        spatial=load_spatial(a.spatial_json,a.strict_spatial)
        art=json.loads((a.data_dir/"artworks_semantics.json").read_text(encoding="utf-8"))
        typed=build_heterogeneous_graph(art["items"],graph,spatial)
        summary=graph_summary(typed)
        put_json(a.data_dir/"heterogeneous_graph.json",typed)
        put_json(a.data_dir/"graph_summary.json",summary)
        put_json(a.data_dir/"spatial_validation.json",spatial["validation"])
        print_graph_summary(summary)
        print(json.dumps({"audit":{k:audit[k] for k in ("visits","raw_starts","raw_ends","paired","unpaired_starts","offline_imputed","offline_imputation_at_gap_cap","availability_conflicts")},
                          "split_sizes":{k:len(v) for k,v in split.items()},"transition_edges":len(graph["transition_edges"])},ensure_ascii=False,indent=2))
    elif a.command=="validate-spatial":
        spatial=load_spatial(a.spatial_json,a.strict)
        put_json(a.output,spatial["validation"])
    elif a.command=="build-graph":
        _,art,_,_,graph=load_processed(a.data_dir,a.split_seed)
        spatial=load_spatial(a.spatial_json,a.strict_spatial)
        typed=build_heterogeneous_graph(art["items"],graph,spatial,a.variant,a.room_features=="metadata",a.midpoint,a.gnn_layers)
        summary=graph_summary(typed)
        put_json(a.output/"graph_train.json",typed)
        put_json(a.output/"graph_summary.json",summary)
        print_graph_summary(summary)
    elif a.command=="baselines":
        results=baselines(a.data_dir,a.output,a.seed,a.history,a.bootstrap,a.availability_mask)
        print(json.dumps({k:{fold:round(results[k][fold]["micro"]["mrr"],4) for fold in ("validation","test")} for k in ("popularity","markov1","markov2")},indent=2))
    elif a.command=="train":
        metrics=train(data_dir=a.data_dir,output=a.output,variant=a.variant,architecture=a.architecture,seed=a.seed,split_seed=a.split_seed,
                      hidden=a.hidden,epochs=a.epochs,patience=a.patience,batch_size=a.batch_size,history=a.history,
                      lr=a.lr,weight_decay=a.weight_decay,dropout=a.dropout,edge_dropout=a.edge_dropout,
                      seq_weight=a.seq_weight,graph_weight=a.graph_weight,temperature=a.temperature,
                      augmentation=not a.no_augmentation,bootstrap=a.bootstrap,mask=a.availability_mask,
                      spatial_json=a.spatial_json,gnn_layers=a.gnn_layers,room_features=a.room_features=="metadata",
                      midpoint=a.midpoint,strict_spatial=a.strict_spatial)
        print(json.dumps({"selected_epoch":metrics["selected_epoch"],"validation_mrr":metrics["validation"]["micro"]["mrr"],"test_status":metrics["test"]},indent=2))
    elif a.command=="evaluate-test":
        result=evaluate_test(a.data_dir,a.run_dir,a.bootstrap)
        print(json.dumps({"test_mrr":result["test"]["micro"]["mrr"],"status":result["status"]},indent=2))
    else:
        print(json.dumps(inference(a.data_dir,a.run_dir,a.visit_id,a.event_index,a.topk),ensure_ascii=False,indent=2))

if __name__=="__main__":main()
