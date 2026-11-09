"""Explicit POI/ROOM/SUBJECT graph and reproducible structural ablations."""
from __future__ import annotations

import copy
import math
import warnings
from collections import Counter

from .spatial import components, room_adjacency_edges, room_nodes, standardization

HETERO_VARIANTS = ("poi_only", "poi_room", "poi_room_spatial", "poi_subject", "poi_room_subject")


def ablation_options(variant):
    if variant not in HETERO_VARIANTS:
        raise ValueError(f"Unknown heterogeneous variant: {variant}")
    return {"room": variant in ("poi_room", "poi_room_spatial", "poi_room_subject"),
            "subject": variant in ("poi_subject", "poi_room_subject"),
            "spatial": variant in ("poi_room_spatial", "poi_room_subject")}


def load_poi_nodes(items, poi_ids):
    """Copy metadata, promoting title/artist from the legacy archival compartment."""
    if set(items) != set(poi_ids) or len(set(poi_ids)) != len(poi_ids):
        raise AssertionError("POI catalogue and graph vocabulary differ")
    nodes = []
    for poi in poi_ids:
        attrs = copy.deepcopy(items[poi])
        legacy = attrs.get("reserved_for_plm_not_model_features", {})
        attrs["title"] = attrs.get("title", legacy.get("title"))
        attrs["artist"] = attrs.get("artist", legacy.get("artist"))
        nodes.append({"id": poi, **attrs})
    return nodes


def load_subject_nodes(items):
    return [{"id": s} for s in sorted({s for v in items.values()
            for s in [v.get("subject_main"), *v.get("subject_secondary", [])] if s})]


def poi_room_relations(items, known_rooms):
    forward, inverse, missing = [], [], []
    for poi, item in sorted(items.items()):
        room = item.get("room_id")
        if room not in known_rooms:
            missing.append(poi)
            continue
        forward.append({"from": poi, "to": room})
        inverse.append({"from": room, "to": poi})
    if missing:
        warnings.warn(f"POI without known ROOM: {missing}; no room inferred", UserWarning, stacklevel=2)
    return forward, inverse, missing


def poi_subject_relations(items):
    main, secondary = [], []
    for poi, item in sorted(items.items()):
        subject = item.get("subject_main")
        if subject:
            main.append({"from": poi, "to": subject})
        for s in sorted(set(item.get("subject_secondary", [])) - {subject, None, ""}):
            secondary.append({"from": poi, "to": s})
    return main, secondary


def transition_normalization(graph):
    """No trajectories are accepted here: input is the checked train manifest."""
    if graph.get("derived_only_from_split") != "train" or not graph.get("train_visitors"):
        raise AssertionError("Behavioral graph must have train-only provenance")
    names = graph["temporal_edge_features"]
    rows = [[e[k] for k in names] for e in graph["transition_edges"]]
    if any(not math.isfinite(v) for row in rows for v in row):
        raise ValueError("Nonfinite train temporal edge statistic")
    norms = [standardization([r[k] for r in rows]) for k in range(len(names))]
    return {"features": names, "mean": [n["mean"] for n in norms], "std": [n["std"] for n in norms],
            "fitted_from": "train_transition_edges_only", "train_visitors": graph["train_visitors"]}


def build_heterogeneous_graph(items, behavior_graph, spatial, variant="poi_room_subject",
                              room_features=True, midpoint=False, layers=2):
    if layers not in (1, 2):
        raise ValueError("Use one or two message-passing layers")
    options = ablation_options(variant)
    if layers == 1 and options["spatial"]:
        raise ValueError("Spatial information needs two layers: ROOM -> ROOM -> POI; use --gnn-layers 2")
    if midpoint and not options["spatial"]:
        raise ValueError("Midpoint requires an ablation with ROOM adjacency")
    if not room_features and not options["room"]:
        raise ValueError("Disabling ROOM features requires an ablation with ROOM nodes")
    nodes = {"POI": load_poi_nodes(items, behavior_graph["poi_ids"])}
    room_data, room_norm = room_nodes(spatial)
    adjacency, midpoint_norm = room_adjacency_edges(spatial)
    main, secondary = poi_subject_relations(items)
    located, contains, missing = poi_room_relations(items, set(spatial["rooms"]))
    floor_conflicts = [p for p, v in items.items() if v.get("room_id") in spatial["rooms"]
                       and v.get("floor") is not None and spatial["rooms"][v["room_id"]]["floor"] is not None
                       and v["floor"] != spatial["rooms"][v["room_id"]]["floor"]]
    if floor_conflicts:
        warnings.warn(f"POI/ROOM floor conflicts: {floor_conflicts}; spatial file is authoritative", UserWarning, stacklevel=2)
    relations = []

    def relation(source, name, target, edges):
        relations.append({"source_type": source, "name": name, "target_type": target, "edges": edges})

    relation("POI", "next_poi", "POI", copy.deepcopy(behavior_graph["transition_edges"]))
    if options["room"]:
        nodes["ROOM"] = room_data
        relation("POI", "located_in", "ROOM", located)
        relation("ROOM", "contains", "POI", contains)
    if options["subject"]:
        nodes["SUBJECT"] = load_subject_nodes(items)
        for name, inverse, edges in [("has_main_subject", "main_subject_of", main),
                                      ("has_secondary_subject", "secondary_subject_of", secondary)]:
            relation("POI", name, "SUBJECT", edges)
            relation("SUBJECT", inverse, "POI", [{"from": e["to"], "to": e["from"]} for e in edges])
    if options["spatial"]:
        for kind in sorted({e["type"] for e in adjacency}):
            relation("ROOM", "adjacent_" + kind, "ROOM", [e for e in adjacency if e["type"] == kind])
    graph = copy.deepcopy(behavior_graph)
    graph.update(format="stkg.cll.heterogeneous.v1", variant=variant, nodes=nodes, relations=relations,
                 room_ids=[n["id"] for n in nodes.get("ROOM", [])],
                 subject_ids=[n["id"] for n in nodes.get("SUBJECT", [])],
                 options={**options, "room_features": bool(room_features and options["room"]),
                          "midpoint": midpoint, "layers": layers},
                 static_relations=[r["name"] for r in relations if r["name"] != "next_poi"],
                 behavior_relations=["next_poi"],
                 spatial_source_sha256=spatial.get("source_sha256"),
                 spatial_validation=spatial["validation"],
                 normalization={"room": room_norm, "geometry": midpoint_norm,
                                "behavior": transition_normalization(behavior_graph)},
                 diagnostic={"poi_without_room": missing, "poi_room_floor_conflicts": floor_conflicts,
                             "poi_without_subject": [p for p, a in sorted(items.items())
                                                      if not a.get("subject_main") and not a.get("subject_secondary")]},
                 behavior_message_directions=["incoming at target", "outgoing neighborhood at source (legacy)"])
    validate_relations(graph)
    return graph


def validate_relations(graph):
    """Check every endpoint and duplication in the final typed graph."""
    ids = {t: {n["id"] for n in nodes} for t, nodes in graph["nodes"].items()}
    for relation in graph["relations"]:
        seen = set()
        for edge in relation["edges"]:
            pair = edge["from"], edge["to"]
            if pair in seen:
                raise AssertionError(f"Duplicate edge: {relation['name']} {pair}")
            seen.add(pair)
            if pair[0] not in ids[relation["source_type"]] or pair[1] not in ids[relation["target_type"]]:
                raise AssertionError(f"Unknown endpoint: {relation['name']} {pair}")
            if relation["name"].startswith("adjacent_") and pair[0] == pair[1]:
                raise AssertionError("No physical adjacency self-loops")


def graph_summary(graph, edge_dropout=0.1):
    adjacency = [e for r in graph["relations"] if r["name"].startswith("adjacent_") for e in r["edges"]]
    room_ids = graph["room_ids"]
    degree = Counter(e["from"] for e in adjacency)
    return {"format": "stkg.cll.graph_summary.v1", "variant": graph["variant"],
            "node_counts": {t: len(n) for t, n in graph["nodes"].items()},
            "edge_counts": {f"{r['source_type']} -> {r['name']} -> {r['target_type']}": len(r["edges"])
                            for r in graph["relations"]},
            "total_edges": sum(len(r["edges"]) for r in graph["relations"]),
            "behavioral_edges": len(graph["transition_edges"]),
            "observed_train_transitions": sum(e["count"] for e in graph["transition_edges"]),
            "train_visitors": graph["train_visitors"], "options": graph["options"],
            "rooms_by_floor": dict(sorted(Counter(n["floor_category"] for n in graph["nodes"].get("ROOM", [])).items())),
            "room_neighbor_count_distribution": dict(sorted(Counter(str(degree[r]) for r in room_ids).items())),
            "isolated_rooms_in_active_graph": [r for r in room_ids if not degree[r]],
            "room_components_in_active_graph": components(room_ids, [(e["from"], e["to"]) for e in adjacency]),
            **graph["diagnostic"], "spatial_file_validation": graph["spatial_validation"],
            "normalization": graph["normalization"], "spatial_source_sha256": graph["spatial_source_sha256"],
            "augmentation": {"edge_dropout": {"POI -> next_poi -> POI": edge_dropout},
                             "static_edge_dropout": 0.0, "representation_dropout": "configured separately"},
            "model_inputs_exclude": ["title", "artist", "visual_description", "style_raw", "annotation_provenance"],
            "message_passing": "Synchronous relational means, shared source projections and per-relation diagonal scales; train frequency/time weighted incoming and outgoing POI messages",
            "receptive_field": ("2 layers: POI -> ROOM/SUBJECT -> POI and adjacent ROOM -> ROOM -> POI; adjacent-room POI would require 3 hops"
                                if graph["options"]["layers"] == 2 else
                                "1 layer: direct ROOM/SUBJECT/behavior neighbors only; spatial and peer-POI paths need 2 layers")}


def print_graph_summary(summary):
    print("NODE TYPES\n----------")
    for kind, count in summary["node_counts"].items():
        print(f"{kind}: {count}")
    print("EDGE TYPES\n----------")
    for kind, count in summary["edge_counts"].items():
        print(f"{kind}: {count}")
    for key in ["rooms_by_floor", "room_neighbor_count_distribution", "isolated_rooms_in_active_graph",
                "poi_without_room", "poi_without_subject"]:
        print(f"{key}: {summary[key]}")
    print("Active ROOM components:", len(summary["room_components_in_active_graph"]))
