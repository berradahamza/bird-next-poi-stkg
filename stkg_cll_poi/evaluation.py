"""Visitor-clustered ranking metrics with a fixed full-POI candidate universe."""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def ranking_records(scores: np.ndarray, targets: np.ndarray, visitor_ids: list[str], poi_ids: list[str]) -> list[dict]:
    if scores.shape!=(len(targets),len(poi_ids)):
        raise ValueError("Expected one full-catalogue score vector per example")
    ordered=np.argsort(-scores,axis=1,kind="stable")
    result=[]
    for n,(ranking,t,v) in enumerate(zip(ordered,targets,visitor_ids)):
        rank=int(np.flatnonzero(ranking==t)[0])+1
        result.append({"visitor":v,"target":poi_ids[t],"rank":rank,
                       "top5":[poi_ids[x] for x in ranking[:5]]})
    return result


def _metrics(records: list[dict]) -> dict:
    r=np.array([v["rank"] for v in records],dtype=float)
    return {"examples":len(r),"acc1":float(np.mean(r==1)),"hit5":float(np.mean(r<=5)),
            "hit10":float(np.mean(r<=10)),"mrr":float(np.mean(1/r))}


def summarize(records: list[dict], bootstrap: int = 1000, seed: int = 2022) -> dict:
    if not records:
        raise ValueError("No ranking records")
    by=defaultdict(list)
    for r in records:by[r["visitor"]].append(r)
    micro=_metrics(records)
    per={k:_metrics(v) for k,v in sorted(by.items())}
    keys=("acc1","hit5","hit10","mrr")
    macro={k:float(np.mean([p[k] for p in per.values()])) for k in keys}
    visitors=sorted(by)
    rng=np.random.default_rng(seed)
    estimates={k:[] for k in keys}
    macro_estimates={k:[] for k in keys}
    for _ in range(bootstrap):
        chosen=rng.choice(visitors,len(visitors),replace=True)
        sample=[r for v in chosen for r in by[v]]
        value=_metrics(sample)
        for k in keys:estimates[k].append(value[k])
        for k in keys:macro_estimates[k].append(float(np.mean([per[v][k] for v in chosen])))
    confidence={k:[float(np.percentile(estimates[k],2.5)),float(np.percentile(estimates[k],97.5))] for k in keys} if bootstrap else {}
    macro_confidence={k:[float(np.percentile(macro_estimates[k],2.5)),float(np.percentile(macro_estimates[k],97.5))] for k in keys} if bootstrap else {}
    return {"micro":micro,"macro_by_visitor":macro,"visitor_count":len(visitors),
            "micro_cluster_bootstrap_95pct":confidence,"macro_cluster_bootstrap_95pct":macro_confidence,
            "per_visitor":per,
            "bootstrap_resamples":bootstrap}
