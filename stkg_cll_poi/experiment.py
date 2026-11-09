"""Executable baselines, training, one-time test, and prefix inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .evaluation import ranking_records,summarize
from .features import examples_by_split,graph_manifest,prefix_features,save_split_and_graph,visitor_split
from .model import CausalTransformerBaseline,STKGCLL,bidirectional_infonce,node_view_loss,sequence_negative_mask
from .preprocess import prepare,put_json,sha256
from .heterogeneous import HETERO_VARIANTS, build_heterogeneous_graph, graph_summary, print_graph_summary
from .spatial import DEFAULT_ADJACENCY, load_spatial


def load_processed(data_dir: Path, seed: int):
    data_dir=Path(data_dir)
    paths={k:data_dir/k for k in ["trajectories_poi.json","artworks_semantics.json","data_audit.json"]}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(f"Run `python -m stkg_cll_poi prepare` first: {p}")
    traj=json.loads(paths["trajectories_poi.json"].read_text(encoding="utf-8"))
    art=json.loads(paths["artworks_semantics.json"].read_text(encoding="utf-8"))
    audit=json.loads(paths["data_audit.json"].read_text(encoding="utf-8"))
    if traj["format"]!="stkg.cll.poi.v1" or art["format"]!="stkg.cll.artworks.v1":
        raise ValueError("Incompatible processed data version")
    split=visitor_split(traj["visitors"],seed)
    graph=graph_manifest(traj,art,split)
    return traj,art,audit,split,graph


def availability(poi_ids:list[str], items:dict, date:str, enabled:bool)->np.ndarray:
    if not enabled:return np.ones(len(poi_ids),dtype=bool)
    return np.array([(not items[p]["hook_date"] or items[p]["hook_date"]<=date) and
                     (not items[p]["unhook_date"] or date<items[p]["unhook_date"])
                     for p in poi_ids],dtype=bool)


def check_mask(audit:dict, enabled:bool):
    if enabled and audit["availability_conflicts"]:
        raise ValueError("Availability catalogue contradicts an observed POI; correct its source metadata and rerun prepare before enabling a mask")


def score_records(scores:np.ndarray, examples:list[dict],poi_ids:list[str],items:dict,enabled:bool):
    index={p:i for i,p in enumerate(poi_ids)}
    for j,e in enumerate(examples):
        mask=availability(poi_ids,items,e["visit_date"],enabled)
        if not mask[index[e["target"]]]:
            raise ValueError("An observed next POI was incorrectly masked")
        scores[j,~mask]=-np.inf
    return ranking_records(scores,np.array([index[e["target"]] for e in examples]),[e["visitor"] for e in examples],poi_ids)


def baseline_scores(examples:list[dict],poi_ids:list[str],pop:Counter,one:dict,two:dict,method:str):
    n=len(poi_ids)
    pos={p:i for i,p in enumerate(poi_ids)}
    def probability(counts:Counter,backoff:np.ndarray):
        if not counts:return backoff
        arr=backoff*0.001
        for p,c in counts.items():arr[pos[p]]+=c
        return arr/arr.sum()
    base=np.array([pop.get(p,0)+0.001 for p in poi_ids],dtype=float)
    base/=base.sum()
    scores=np.empty((len(examples),n))
    for i,e in enumerate(examples):
        h=e["poi_history"]
        if method=="popularity":v=base
        elif method=="markov1":v=probability(one.get(h[-1]),base)
        elif method=="markov2":
            prev=probability(one.get(h[-1]),base)
            v=probability(two.get(tuple(h[-2:])),prev) if len(h)>=2 else prev
        else:raise ValueError(method)
        scores[i]=v
    return scores


def baselines(data_dir:Path, output:Path, seed:int=2022, max_history:int=32,bootstrap:int=1000, mask:bool=False):
    traj,art,audit,split,graph=load_processed(data_dir,seed)
    check_mask(audit,mask)
    ex=examples_by_split(traj,split,max_history)
    pop=Counter();one=defaultdict(Counter);two=defaultdict(Counter)
    for vid in split["train"]:
        seq=[e["poi_id"] for e in traj["visitors"][vid]["events"]]
        for i in range(1,len(seq)):
            pop[seq[i]]+=1;one[seq[i-1]][seq[i]]+=1
            if i>=2:two[(seq[i-2],seq[i-1])][seq[i]]+=1
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results={"scope":"full-catalogue exact ranking; train-only behavioral counts", "seed":seed,
             "train_visitors":split["train"], "split":split,
             "candidate_mask":mask,"smoothing":"0.001 times backoff for seen contexts; 0.001 raw popularity"}
    for method in ("popularity","markov1","markov2"):
        results[method]={}
        for fold in ("validation","test"):
            scores=baseline_scores(ex[fold],graph["poi_ids"],pop,one,two,method)
            records=score_records(scores,ex[fold],graph["poi_ids"],art["items"],mask)
            results[method][fold]=summarize(records,bootstrap,seed)
    put_json(output/"baselines.json",results)
    return results


def seed_all(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)


def make_model(config:dict,art:dict,graph:dict):
    arch=config["architecture"]
    if arch=="gru":
        return STKGCLL(art["items"],graph,config["hidden"],config["dropout"],
                       config["edge_dropout"] if config["augmentation"] else 0,config["variant"])
    if arch in ("sasrec_style","temporal_transformer"):
        return CausalTransformerBaseline(len(graph["poi_ids"]),config["hidden"],
                                         config["max_history"],config["dropout"],arch=="temporal_transformer")
    raise ValueError(f"Unsupported architecture {arch}")


def fit_feature_normalizer(examples:list[dict], train_visitors:list[str]|None=None):
    if train_visitors is None or not examples or any(e["visitor"] not in set(train_visitors) for e in examples):
        raise AssertionError("Feature normalization requires explicitly declared train-only prefixes")
    arr=np.array([row for e in examples for row in e["temporal"]],dtype=float)
    mu=np.zeros(5);sigma=np.ones(5)
    for k in (0,1,4):
        mu[k]=arr[:,k].mean();sigma[k]=max(arr[:,k].std(),1e-6)
    return {"mean":mu.tolist(),"std":sigma.tolist(),"fitted_from":"train_prefixes_only", "train_visitors":list(train_visitors)}


def load_run_graph(run_dir:Path, config:dict, rebuilt_behavior:dict):
    """Restore the frozen typed graph, never the user's subsequently edited plan."""
    if config["variant"] not in HETERO_VARIANTS:
        return rebuilt_behavior
    for filename, digest in config["run_input_sha256"].items():
        if sha256(run_dir/filename) != digest:
            raise ValueError(f"Frozen run input differs from checkpoint lineage: {filename}")
    graph=json.loads((run_dir/"graph_train.json").read_text(encoding="utf-8"))
    for field in ["poi_ids","transition_edges","train_visitors","train_target_counts"]:
        if graph[field] != rebuilt_behavior[field]:
            raise AssertionError(f"Frozen behavior graph differs from the train-only reconstruction: {field}")
    return graph


def batches(examples:list[dict],item_index:dict,batch_size:int,normalizer:dict,shuffle:bool,rng:random.Random):
    idx=list(range(len(examples)))
    if shuffle:rng.shuffle(idx)
    mu=np.array(normalizer["mean"],dtype=np.float32);sd=np.array(normalizer["std"],dtype=np.float32)
    for k in range(0,len(idx),batch_size):
        chunk=[examples[j] for j in idx[k:k+batch_size]]
        length=np.array([len(e["poi_history"]) for e in chunk],dtype=np.int64)
        x=np.zeros((len(chunk),int(length.max())),dtype=np.int64)
        t=np.zeros((len(chunk),x.shape[1],5),dtype=np.float32)
        y=np.array([item_index[e["target"]] for e in chunk],dtype=np.int64)
        for i,e in enumerate(chunk):
            n=length[i];x[i,:n]=[item_index[p] for p in e["poi_history"]]
            t[i,:n]=(np.array(e["temporal"],dtype=np.float32)-mu)/sd
        yield chunk,torch.from_numpy(x),torch.from_numpy(t),torch.from_numpy(length),torch.from_numpy(y)


def evaluate_model(model:STKGCLL, examples:list[dict],item_index:dict,normalizer:dict,
                   batch_size:int,items:dict,mask:bool,bootstrap:int,seed:int):
    model.eval();rng=random.Random(0);scores=[]
    with torch.no_grad():
        for _,x,t,lengths,_ in batches(examples,item_index,batch_size,normalizer,False,rng):
            logits,_,_=model(x,t,lengths);scores.append(logits.cpu().numpy())
    poi_ids=list(item_index)
    records=score_records(np.concatenate(scores),examples,poi_ids,items,mask)
    return summarize(records,bootstrap,seed),records


def train(data_dir:Path, output:Path,variant:str="poi_room_subject",architecture:str="gru",seed:int=2022,split_seed:int=2022,hidden:int=64,
          epochs:int=30,patience:int=5,batch_size:int=64,history:int=32,
          lr:float=.001,weight_decay:float=.00005,dropout:float=.2,edge_dropout:float=.1,
          seq_weight:float=.05,graph_weight:float=.02,temperature:float=.2,
          augmentation:bool=True,bootstrap:int=1000,mask:bool=False,
          spatial_json:Path=DEFAULT_ADJACENCY,gnn_layers:int=2,room_features:bool=True,
          midpoint:bool=False,strict_spatial:bool=False):
    if min(epochs,patience,batch_size,hidden,history)<1 or seq_weight<0 or graph_weight<0 or temperature<=0:
        raise ValueError("Invalid training configuration")
    if architecture!="gru" and (variant!="poi_id" or seq_weight or graph_weight):
        raise ValueError("Transformer comparison requires variant=poi_id and zero contrast weights")
    if not augmentation and (seq_weight or graph_weight):
        raise ValueError("Disable both contrast weights when disabling augmentation")
    if not 0 <= dropout < 1 or not 0 <= edge_dropout <= 1 or bootstrap < 0:
        raise ValueError("Invalid dropout or bootstrap configuration")
    if variant not in HETERO_VARIANTS and (midpoint or not room_features or gnn_layers!=2):
        raise ValueError("Spatial/GNN options require a heterogeneous variant")
    output=Path(output)
    if any((output/name).exists() for name in ["config.json","model.pt","metrics.json","test_metrics.json"]):
        raise FileExistsError("Run already exists; use a new --output directory to preserve checkpoint/results lineage")
    traj,art,audit,split,graph=load_processed(data_dir,split_seed)
    check_mask(audit,mask)
    if graph["train_visitors"] != split["train"]:
        raise AssertionError("Behavioral graph train visitor mismatch")
    if variant in HETERO_VARIANTS:
        spatial=load_spatial(spatial_json,strict_spatial)
        graph=build_heterogeneous_graph(art["items"],graph,spatial,variant,room_features,midpoint,gnn_layers)
    seed_all(seed)
    output.mkdir(parents=True,exist_ok=True)
    ex=examples_by_split(traj,split,history,True)
    item_index={p:i for i,p in enumerate(graph["poi_ids"])}
    normalizer=fit_feature_normalizer(ex["train"],split["train"])
    config={"variant":variant,"architecture":architecture,"seed":seed,"split_seed":split_seed,"hidden":hidden,"epochs":epochs,"patience":patience,
            "batch_size":batch_size,"max_history":history,"lr":lr,"weight_decay":weight_decay,
            "dropout":dropout,"edge_dropout":edge_dropout,"sequence_contrast_weight":seq_weight,
            "graph_contrast_weight":graph_weight,"temperature":temperature,"augmentation":augmentation,
            "cluster_bootstrap_resamples":bootstrap,"availability_mask":mask,
            "torch_version":str(torch.__version__),"numpy_version":str(np.__version__),
            "processed_input_sha256":{n:sha256(Path(data_dir)/n) for n in ["trajectories_poi.json","artworks_semantics.json"]},
            "test_selection_policy":"best epoch selected only by validation MRR; separate evaluate-test command after freezing configuration"}
    put_json(output/"split.json",{"visitors":split,"split_seed":split_seed})
    put_json(output/"graph_train.json",graph)
    put_json(output/"feature_normalizer.json",normalizer)
    if variant in HETERO_VARIANTS:
        shutil.copyfile(spatial_json,output/"bird_adjacency.json")
        if sha256(output/"bird_adjacency.json") != graph["spatial_source_sha256"]:
            raise ValueError("Spatial source changed while building the graph")
        config.update(graph_encoder="heterogeneous_relational_mean_v1",gnn_layers=gnn_layers,
                      room_features=room_features,midpoint=midpoint,
                      spatial_input_sha256=graph["spatial_source_sha256"],
                      run_input_sha256={n:sha256(output/n) for n in ["graph_train.json","split.json","feature_normalizer.json","bird_adjacency.json"]})
    put_json(output/"config.json",config)
    model=make_model(config,art,graph)
    if variant in HETERO_VARIANTS:
        summary=graph_summary(graph,edge_dropout if augmentation else 0)
        summary["trainable_parameters"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
        summary["augmentation"]["representation_dropout"]=dropout
        put_json(output/"graph_summary.json",summary)
        print_graph_summary(summary)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    rng=random.Random(seed)
    best=-float("inf");wait=0;learning=[];best_epoch=None
    for epoch in range(1,epochs+1):
        model.train();running={"ce":0.0,"seq_cl":0.0,"graph_cl":0.0,"grad_norm_before_clip":0.0};num=0;neg_count=0
        for chunk,x,t,lengths,y in batches(ex["train"],item_index,batch_size,normalizer,True,rng):
            logits1,h1,z1=model(x,t,lengths)
            # Keep the same two stochastic supervised passes when λ=0, so
            # "augmentation without contrast" isolates the auxiliary loss.
            use_two=augmentation
            if use_two:
                logits2,h2,z2=model(x,t,lengths)
                ce=(F.cross_entropy(logits1,y)+F.cross_entropy(logits2,y))/2
                neg=sequence_negative_mask(x,lengths,y,True)
                seq_cl=bidirectional_infonce(model.seq_project(h1),model.seq_project(h2),neg,temperature) if seq_weight else h1.sum()*0
                graph_cl=node_view_loss(model,z1,z2,x,lengths,temperature) if graph_weight else z1.sum()*0
                neg_count+=int(neg.sum().item())
            else:
                ce=F.cross_entropy(logits1,y);seq_cl=h1.sum()*0;graph_cl=z1.sum()*0
            loss=ce+seq_weight*seq_cl+graph_weight*graph_cl
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            n=len(chunk);num+=n
            for key,value in [("ce",ce),("seq_cl",seq_cl),("graph_cl",graph_cl)]:running[key]+=float(value.detach())*n
            running["grad_norm_before_clip"]+=float(grad_norm.detach())*n
        val,_=evaluate_model(model,ex["validation"],item_index,normalizer,batch_size,art["items"],mask,0,seed)
        row={"epoch":epoch,"train_loss_per_example":{k:v/num for k,v in running.items()},
             "valid_negative_entries":neg_count,"valid_negative_entries_per_anchor":neg_count/max(num,1),
             "validation_mrr":val["micro"]["mrr"],
             "validation_acc1":val["micro"]["acc1"]}
        learning.append(row)
        print(f"epoch {epoch:02d} ce={running['ce']/num:.4f} val_mrr={row['validation_mrr']:.4f}",flush=True)
        if val["micro"]["mrr"] > best+1e-9:
            best=val["micro"]["mrr"];best_epoch=epoch;wait=0
            torch.save({"state_dict":model.state_dict(),"config":config,"poi_ids":graph["poi_ids"]},output/"model.pt")
        else:
            wait+=1
            if wait>=patience:break
    # Loading a weights-only checkpoint prevents untrusted pickled object execution.
    checkpoint=torch.load(output/"model.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validation,_=evaluate_model(model,ex["validation"],item_index,normalizer,batch_size,art["items"],mask,bootstrap,seed)
    results={"status":"one seed; exploratory until planned ablations and multiple seeds are run",
             "selected_epoch":best_epoch,"trained_epochs":len(learning),"selection_metric":"validation.micro.mrr",
             "n_examples":{k:len(v) for k,v in ex.items()},"validation":validation,
             "test":"not_evaluated_use_evaluate-test_after_freezing_configuration",
             "history":learning}
    put_json(output/"metrics.json",results)
    return results


def evaluate_test(data_dir:Path,run_dir:Path,bootstrap:int|None=None):
    """Release the sealed test result once; do not select a model using this result."""
    data_dir,run_dir=Path(data_dir),Path(run_dir)
    if (run_dir/"test_metrics.json").exists():
        raise FileExistsError("Test already evaluated in this run directory; immutable result")
    config=json.loads((run_dir/"config.json").read_text())
    for filename,digest in config["processed_input_sha256"].items():
        if sha256(data_dir/filename)!=digest:
            raise ValueError("Processed input differs from checkpoint lineage")
    traj,art,audit,split,graph=load_processed(data_dir,config["split_seed"])
    graph=load_run_graph(run_dir,config,graph)
    seed_all(config["seed"])
    check_mask(audit,config["availability_mask"])
    ex=examples_by_split(traj,split,config["max_history"])
    item_index={p:i for i,p in enumerate(graph["poi_ids"])}
    normalizer=json.loads((run_dir/"feature_normalizer.json").read_text())
    model=make_model(config,art,graph)
    checkpoint=torch.load(run_dir/"model.pt",map_location="cpu",weights_only=True)
    if checkpoint["config"] != config:
        raise ValueError("Run configuration differs from checkpoint")
    if checkpoint["poi_ids"]!=graph["poi_ids"]:
        raise ValueError("POI vocabulary mismatch")
    model.load_state_dict(checkpoint["state_dict"])
    records=None
    result,records=evaluate_model(model,ex["test"],item_index,normalizer,config["batch_size"],
                                  art["items"],config["availability_mask"],
                                  config["cluster_bootstrap_resamples"] if bootstrap is None else bootstrap,config["seed"])
    seen={e["target"] for e in ex["train"]}
    strata={"seen":[],"unseen":[]}
    for record in records:
        strata["seen" if record["target"] in seen else "unseen"].append(record)
    report={"status":"test evaluated after frozen validation checkpoint; one split/seed, exploratory",
            "test":result,"target_seen_in_train":{k:summarize(v,0,config["seed"]) if v else {"examples":0} for k,v in strata.items()},
            "checkpoint_sha256":sha256(run_dir/"model.pt")}
    put_json(run_dir/"test_metrics.json",report)
    with (run_dir/"test_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["visitor","target","rank","top5"])
        for v in records:writer.writerow([v["visitor"],v["target"],v["rank"],";".join(v["top5"])])
    return report


def inference(data_dir:Path,run_dir:Path,visit_id:str,event_index:int,topk:int=5):
    run_dir=Path(run_dir);config=json.loads((run_dir/"config.json").read_text())
    for filename,digest in config["processed_input_sha256"].items():
        if sha256(Path(data_dir)/filename)!=digest:
            raise ValueError("Processed input differs from checkpoint lineage")
    traj,art,audit,split,graph=load_processed(data_dir,config["split_seed"])
    graph=load_run_graph(run_dir,config,graph)
    seed_all(config["seed"])
    check_mask(audit,config["availability_mask"])
    if visit_id not in traj["visitors"]:raise ValueError("Unknown visitor ID")
    events=traj["visitors"][visit_id]["events"]
    poi,temporal=prefix_features(events,event_index,config["max_history"])
    item_index={p:i for i,p in enumerate(graph["poi_ids"])}
    norm=json.loads((run_dir/"feature_normalizer.json").read_text())
    x=torch.tensor([[item_index[p] for p in poi]],dtype=torch.long)
    temporal_array=(np.array(temporal,dtype=np.float32)-np.array(norm["mean"],dtype=np.float32))/np.array(norm["std"],dtype=np.float32)
    t=torch.from_numpy(temporal_array[None,:,:])
    length=torch.tensor([len(poi)])
    model=make_model(config,art,graph)
    checkpoint=torch.load(run_dir/"model.pt",map_location="cpu",weights_only=True)
    if checkpoint["config"] != config:
        raise ValueError("Run configuration differs from checkpoint")
    if checkpoint["poi_ids"]!=graph["poi_ids"]:raise ValueError("POI vocabulary mismatch")
    model.load_state_dict(checkpoint["state_dict"]);model.eval()
    with torch.no_grad():scores=model(x,t,length)[0][0].numpy()
    scores[~availability(graph["poi_ids"],art["items"],traj["visitors"][visit_id]["visit_date_from_filename"],config["availability_mask"])]=-np.inf
    rank=np.argsort(-scores,kind="stable")[:topk]
    return {"visit":visit_id,"prefix_including_event_index":event_index,
            "predictions":[{"poi_id":graph["poi_ids"][j],"logit":float(scores[j])} for j in rank],
            "note":"Only the prefix is passed into the model; the true next POI is not read in this function."}
