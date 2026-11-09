"""Causal prefix generation and visitor-disjoint splitting."""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

from .preprocess import put_json, sha256

FEATURES = ("log_gap_from_previous_start", "log_previous_dwell", "dwell_observed", "dwell_imputed_causally", "log_elapsed_since_first_start")


def assert_visitor_split(trajectories: dict, split: dict):
    """Fail on overlap, duplicates, unknown visitors or an incomplete split."""
    if set(split) != {"train", "validation", "test"}:
        raise AssertionError("Expected train / validation / test")
    all_ids = [v for ids in split.values() for v in ids]
    if any(not ids for ids in split.values()) or len(all_ids) != len(set(all_ids)):
        raise AssertionError("Empty, overlapping or duplicate visitor split")
    if set(all_ids) != set(trajectories["visitors"]):
        raise AssertionError("Split must cover exactly the provided visitors")


def visitor_split(visitors: dict, seed: int = 2022) -> dict[str, list[str]]:
    ids = sorted(visitors)
    random.Random(seed).shuffle(ids)
    held = max(1, round(len(ids)*0.15))
    if len(ids) <= 2*held:
        raise ValueError("Need nonempty train / validation / test")
    return {"train": ids[:-2*held], "validation": ids[-2*held:-held], "test": ids[-held:]}


def causal_duration(events: list[dict], i: int, current_index: int) -> tuple[float, int, int]:
    """Previous event's duration known by start[current_index].

    A missing end can be imputed only after the *following* start is known,
    using only earlier reliable ends available at that following start.
    """
    if i >= current_index:
        return 0.0, 0, 0
    current_time = events[current_index]["t_start_seconds"]
    e = events[i]
    if e["t_end_observed_seconds"] is not None and e["t_end_observed_seconds"] <= current_time+1e-9:
        return e["duration_observed_seconds"], 1, 0
    if e["t_end_observed_seconds"] is not None or i+1 > current_index:
        return 0.0, 0, 0
    gap = events[i+1]["t_start_seconds"]-e["t_start_seconds"]
    cutoff = events[i+1]["t_start_seconds"]
    earlier = [x["duration_observed_seconds"] for x in events[:i]
               if x["duration_observed_seconds"] is not None and x["duration_observed_seconds"] > 0
               and x["t_end_observed_seconds"] <= cutoff+1e-9]
    if not earlier or gap <= 0:
        return 0.0, 0, 0
    return min(gap, (gap+sum(earlier)/len(earlier))/2), 0, 1


def prefix_features(events: list[dict], current_index: int, max_history: int, temporal: bool = True) -> tuple[list[str], list[list[float]]]:
    if current_index >= len(events)-1 or current_index < 0:
        raise ValueError("A prefix requires a next POI target")
    begin = max(0, current_index-max_history+1)
    first = events[0]["t_start_seconds"]
    poi, features = [], []
    for i in range(begin, current_index+1):
        e = events[i]
        poi.append(e["poi_id"])
        gap = e["t_start_seconds"]-events[i-1]["t_start_seconds"] if i else 0.0
        dwell, observed, imputed = causal_duration(events,i,current_index)
        row = [math.log1p(gap), math.log1p(dwell), float(observed), float(imputed),
               math.log1p(e["t_start_seconds"]-first)] if temporal else [0.0]*len(FEATURES)
        features.append(row)
    return poi, features


def examples_by_split(trajectories: dict, split: dict, max_history: int = 32, temporal: bool = True) -> dict:
    assert_visitor_split(trajectories, split)
    all_examples = {}
    for fold, visitor_ids in split.items():
        examples = []
        for visitor_id in visitor_ids:
            events = trajectories["visitors"][visitor_id]["events"]
            for current in range(len(events)-1):
                poi, time = prefix_features(events,current,max_history,temporal)
                examples.append({"visitor": visitor_id, "event_index": current,
                                 "poi_history": poi, "temporal": time,
                                 "target": events[current+1]["poi_id"],
                                 "visit_date": trajectories["visitors"][visitor_id]["visit_date_from_filename"]})
        if not examples:
            raise ValueError(f"No examples in {fold}")
        all_examples[fold] = examples
    return all_examples


def graph_manifest(trajectories: dict, artworks: dict, split: dict) -> dict:
    assert_visitor_split(trajectories, split)
    items = artworks["items"]
    edge_counts = Counter()
    train_targets = Counter()
    edge_times = {}
    for vid in split["train"]:
        events=trajectories["visitors"][vid]["events"]
        sequence = [e["poi_id"] for e in events]
        edge_counts.update(zip(sequence,sequence[1:]))
        train_targets.update(sequence[1:])
        for i,(a,b) in enumerate(zip(events,events[1:])):
            key=(a["poi_id"],b["poi_id"])
            sums=edge_times.setdefault(key,{"log_gap":0.,"log_dwell":0.,"dwell_known":0})
            sums["log_gap"]+=math.log1p(b["t_start_seconds"]-a["t_start_seconds"])
            dwell,observed,imputed=causal_duration(events,i,i+1)
            if observed or imputed:
                sums["log_dwell"]+=math.log1p(dwell)
                sums["dwell_known"]+=1
    rooms = sorted({r["room_id"] for r in items.values()})
    subjects = sorted({s for v in items.values() for s in ([v["subject_main"]] + v["subject_secondary"]) if s is not None})
    edges=[]
    for (a,b),count in sorted(edge_counts.items()):
        time=edge_times[(a,b)]
        edges.append({"from":a,"to":b,"count":count,
                      "mean_log_start_gap":time["log_gap"]/count,
                      "mean_log_causal_dwell":time["log_dwell"]/time["dwell_known"] if time["dwell_known"] else 0.,
                      "causal_dwell_available_rate":time["dwell_known"]/count})
    expected = sum(len(trajectories["visitors"][v]["events"])-1 for v in split["train"])
    if sum(e["count"] for e in edges) != expected or sum(train_targets.values()) != expected:
        raise AssertionError("Behavior counts must equal train-only consecutive transitions")
    return {"format":"stkg.cll.graph.v1", "derived_only_from_split":"train",
            "poi_ids": sorted(items), "room_ids": rooms, "subject_ids": subjects,
            "transition_edges": edges,
            "temporal_edge_features": ["mean_log_start_gap","mean_log_causal_dwell","causal_dwell_available_rate"],
            "train_target_counts": dict(sorted(train_targets.items())),
            "static_relations": ["in_room","main_subject","secondary_subject"],
            "behavior_relations": ["transition_out","transition_in"],
            "feature_exclusions": ["visual_description","style_raw","artist","title"],
            "train_visitors": split["train"]}


def save_split_and_graph(processed: Path, seed=2022) -> tuple[dict, dict]:
    processed=Path(processed)
    traj=json.loads((processed/"trajectories_poi.json").read_text(encoding="utf-8"))
    artworks=json.loads((processed/"artworks_semantics.json").read_text(encoding="utf-8"))
    split=visitor_split(traj["visitors"],seed)
    if len(set(split["train"]) & set(split["validation"])) or len(set(split["train"]) & set(split["test"])) or len(set(split["test"]) & set(split["validation"])):
        raise AssertionError("Visitor split overlap")
    graph=graph_manifest(traj,artworks,split)
    put_json(processed/"split.json",{"format":"stkg.cll.split.v1","seed":seed,"visitors":split})
    put_json(processed/"graph_train.json",graph)
    return split,graph
