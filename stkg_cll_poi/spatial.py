"""Validate the supplied BIRD plan; never infer doors or vertical connections."""
from __future__ import annotations

import copy
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from .preprocess import sha256

DEFAULT_ADJACENCY = Path(__file__).resolve().parent.parent / "data/raw/bird_adjacency.json"


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def components(room_ids, pairs):
    """Connected components including empty rooms and isolated rooms."""
    neighbors = {r: set() for r in room_ids}
    for a, b in pairs:
        neighbors[a].add(b)
        neighbors[b].add(a)
    unseen, result = set(room_ids), []
    while unseen:
        pending, found = [min(unseen)], set()
        while pending:
            room = pending.pop()
            if room in found:
                continue
            found.add(room)
            pending.extend(neighbors[room] - found)
        unseen -= found
        result.append(sorted(found))
    return result


def validate_spatial(data: dict, strict: bool = False, emit_warnings: bool = True):
    """Return sanitized rooms/pairs and an exhaustive diagnostic.

    Reciprocal declarations in neighbors are authoritative. One-sided, unknown
    and self references are excluded, with warnings. Missing/conflicting edge
    metadata become type=unknown, never invented geometry. Strict mode rejects
    any issue after reporting it. All normalizations concern static data only.
    """
    if not isinstance(data, dict) or not isinstance(data.get("rooms"), dict) or not data["rooms"]:
        raise ValueError("Spatial JSON requires a nonempty rooms object")
    issues = []

    def issue(code, detail):
        issues.append({"code": code, "detail": detail})

    rooms = {}
    for room, raw in sorted(data["rooms"].items()):
        raw = raw if isinstance(raw, dict) else {}
        floor, area = raw.get("floor"), raw.get("area_m2")
        if not finite_number(floor) or int(floor) != floor:
            issue("invalid_floor", room)
            floor = None
        else:
            floor = int(floor)
        if not finite_number(area) or area <= 0:
            issue("invalid_area_m2", room)
            area = None
        rooms[room] = {"floor": floor, "area_m2": area}

    raw_neighbors = data.get("neighbors")
    if not isinstance(raw_neighbors, dict):
        issue("missing_neighbors", "No usable topology; no spatial edges constructed")
        raw_neighbors = {}
    declarations = set()
    for a, neighbors in sorted(raw_neighbors.items()):
        if a not in rooms:
            issue("invalid_room_reference", str(a))
            continue
        if not isinstance(neighbors, list):
            issue("invalid_neighbors_list", a)
            continue
        seen = set()
        for b in neighbors:
            if not isinstance(b, str) or b not in rooms:
                issue("invalid_room_reference", f"neighbors[{a}] -> {b!r}")
                continue
            if a == b:
                issue("self_loop", a)
                continue
            if b in seen:
                issue("duplicate_neighbor", f"{a} -> {b}")
            seen.add(b)
            declarations.add((a, b))
    for room in sorted(set(rooms) - set(raw_neighbors)):
        issue("missing_neighbor_entry", room)
    for a, b in sorted(declarations):
        if (b, a) not in declarations:
            issue("asymmetric_neighbors", f"{a} -> {b}; reverse absent, pair excluded")
    topology = {tuple(sorted((a, b))) for a, b in declarations if (b, a) in declarations}

    records = defaultdict(list)
    raw_edges = data.get("adjacencies", [])
    if not isinstance(raw_edges, list):
        issue("invalid_adjacencies", "Expected list")
        raw_edges = []
    for index, raw in enumerate(raw_edges):
        if not isinstance(raw, dict) or not isinstance(raw.get("rooms"), list) or len(raw["rooms"]) != 2:
            issue("invalid_pair", f"adjacencies[{index}]")
            continue
        a, b = raw["rooms"]
        if not isinstance(a, str) or not isinstance(b, str) or a not in rooms or b not in rooms:
            issue("invalid_room_reference", f"adjacencies[{index}] {a!r}, {b!r}")
            continue
        if a == b:
            issue("self_loop", f"adjacencies[{index}] {a}")
            continue
        pair = tuple(sorted((a, b)))
        if (a, b) not in declarations or (b, a) not in declarations:
            issue("adjacency_missing_in_neighbors", f"{pair}; excluded from spatial topology")
        kind = raw.get("type")
        if not isinstance(kind, str) or not kind.strip():
            issue("invalid_edge_type", str(pair))
            kind = "unknown"
        fa, fb = rooms[a]["floor"], rooms[b]["floor"]
        actual_same = fa == fb if fa is not None and fb is not None else None
        declared_same = raw.get("same_floor_rooms")
        if not isinstance(declared_same, bool):
            issue("invalid_same_floor_rooms", str(pair))
        elif actual_same is not None and actual_same != declared_same:
            issue("inconsistent_same_floor_rooms", str(pair))
        same = actual_same if actual_same is not None else declared_same if isinstance(declared_same, bool) else None
        if "floor" in raw and (not finite_number(raw["floor"]) or (actual_same is True and raw["floor"] != fa)):
            issue("invalid_edge_floor", str(pair))
        midpoint = raw.get("midpoint")
        if not isinstance(midpoint, list) or len(midpoint) != 2 or not all(finite_number(x) for x in midpoint):
            issue("missing_or_invalid_midpoint", str(pair))
            midpoint = None
        points = raw.get("points")
        if points is not None and (not isinstance(points, list) or any(
                not isinstance(p, list) or len(p) != 2 or not all(finite_number(x) for x in p) for p in points)):
            issue("invalid_points", str(pair))
            points = None
        records[pair].append({"rooms": list(pair), "type": kind,
                              "same_floor_rooms": same, "floor": fa if actual_same is True else None,
                              "midpoint": midpoint, "points": points, "anchors": raw.get("anchors", [])})

    edges = []
    for pair in sorted(set(records) | topology):
        entries = records.get(pair, [])
        if len(entries) > 1:
            fingerprints = {json.dumps(e, sort_keys=True, ensure_ascii=False) for e in entries}
            code = "duplicate_adjacency" if len(fingerprints) == 1 else "conflicting_duplicate_adjacency"
            issue(code, str(pair))
            if len(fingerprints) > 1:
                entries = []  # Preserve the declared pair, but do not choose arbitrary attributes.
        if pair not in topology:
            continue
        if not entries:
            if pair not in records:
                issue("neighbors_missing_adjacency_attributes", str(pair))
            fa, fb = (rooms[x]["floor"] for x in pair)
            same = fa == fb if fa is not None and fb is not None else None
            entries = [{"rooms": list(pair), "type": "unknown", "same_floor_rooms": same,
                        "floor": fa if same else None, "midpoint": None, "points": None, "anchors": []}]
        edges.append(entries[0])

    degree = Counter(r for p in topology for r in p)
    connected = components(rooms, topology)
    cross = [e["rooms"] for e in edges if e["same_floor_rooms"] is False]
    report = {"rooms": len(rooms), "rooms_by_floor": dict(sorted(Counter(
                  str(v["floor"]) if v["floor"] is not None else "unknown" for v in rooms.values()).items())),
              "adjacency_pairs": len(edges), "directed_adjacency_edges": 2 * len(edges),
              "adjacency_types": dict(sorted(Counter(e["type"] for e in edges).items())),
              "rooms_without_neighbors": [r for r in rooms if not degree[r]],
              "neighbor_count_distribution": dict(sorted(Counter(str(degree[r]) for r in rooms).items())),
              "cross_floor_edges": cross, "connected_components": connected,
              "connected_component_count": len(connected), "issues": issues,
              "issue_counts": dict(sorted(Counter(i["code"] for i in issues).items())),
              "invalid_room_references": sum(i["code"] == "invalid_room_reference" for i in issues),
              "asymmetric_neighbor_declarations": sum(i["code"] == "asymmetric_neighbors" for i in issues),
              "topology_policy": "reciprocal neighbors only; no inferred self/vertical edges",
              "spatial_limit": "No path between distinct room components; behavioral POI transitions may still cross floors." if len(connected) > 1 else None}
    if emit_warnings:
        for i in issues:
            warnings.warn(f"Spatial graph [{i['code']}]: {i['detail']}", UserWarning, stacklevel=2)
    if strict and issues:
        raise ValueError(f"Spatial validation failed: {len(issues)} issue(s); see warnings")
    return {"rooms": rooms, "adjacencies": edges, "validation": report,
            "units": data.get("units"), "source_metadata": copy.deepcopy(data.get("sources", {}))}


def load_spatial(path: Path = DEFAULT_ADJACENCY, strict: bool = False, verbose: bool = True):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    spatial = validate_spatial(raw, strict)
    spatial["source_sha256"] = sha256(path)
    if verbose:
        print_spatial_validation(spatial["validation"])
    return spatial


def print_spatial_validation(report):
    print("Spatial graph validation\n------------------------")
    for label, key in [("Rooms", "rooms"), ("Floors / room counts", "rooms_by_floor"),
                       ("Adjacency pairs", "adjacency_pairs"), ("Directed adjacency edges after expansion", "directed_adjacency_edges"),
                       ("Adjacency types", "adjacency_types"), ("Rooms without neighbors", "rooms_without_neighbors"),
                       ("Invalid room references", "invalid_room_references"), ("Asymmetric neighbor declarations", "asymmetric_neighbor_declarations"),
                       ("Cross-floor edges", "cross_floor_edges"), ("Connected components", "connected_component_count")]:
        print(f"{label}: {report[key]}")
    if report["spatial_limit"]:
        print(report["spatial_limit"])


def standardization(values):
    if not values:
        return {"mean": 0.0, "std": 1.0, "n": 0}
    mean = sum(values) / len(values)
    std = max(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)), 1e-6)
    return {"mean": mean, "std": std, "n": len(values)}


def room_nodes(spatial):
    """Categorical room/floor IDs and a z-score of static room area."""
    norm = standardization([r["area_m2"] for r in spatial["rooms"].values() if r["area_m2"] is not None])
    nodes = []
    for room, data in spatial["rooms"].items():
        area = data["area_m2"]
        nodes.append({"id": room, **data,
                      "floor_category": str(data["floor"]) if data["floor"] is not None else "unknown",
                      "area_normalized": (area - norm["mean"]) / norm["std"] if area is not None else 0.0,
                      "area_available": area is not None})
    return nodes, {"area_m2": norm, "fitted_from": "all_static_museum_rooms"}


def room_adjacency_edges(spatial):
    """Expand reciprocal topology; midpoint z-scores use each floor's own frame."""
    by_floor = defaultdict(list)
    for e in spatial["adjacencies"]:
        if e["same_floor_rooms"] is True and e["floor"] is not None and e["midpoint"] is not None:
            by_floor[str(e["floor"])].append(e["midpoint"])
    norms = {floor: [standardization([p[k] for p in points]) for k in range(2)]
             for floor, points in sorted(by_floor.items())}
    result = []
    for e in spatial["adjacencies"]:
        floor = str(e["floor"])
        available = e["midpoint"] is not None and floor in norms and e["same_floor_rooms"] is True
        midpoint = [(e["midpoint"][k] - norms[floor][k]["mean"]) / norms[floor][k]["std"]
                    for k in range(2)] if available else [0.0, 0.0]
        a, b = e["rooms"]
        attrs = {k: copy.deepcopy(v) for k, v in e.items() if k != "rooms"}
        attrs.update(midpoint_normalized=midpoint, midpoint_available=available)
        result.extend({"from": src, "to": dst, **attrs} for src, dst in [(a, b), (b, a)])
    return result, {"midpoint_by_floor": norms, "fitted_from": "static_same_floor_adjacencies",
                    "cross_floor_geometry": "disabled; coordinate systems are not compared"}
