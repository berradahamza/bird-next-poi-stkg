"""Deterministic, auditable conversion from BIRD observation logs to POI data."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

VERSION = "stkg.cll.poi.v1"
BAD_SUBJECTS = {"", "NA", "N/A", "NONE", "NULL"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def put_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def subject(value: str) -> str | None:
    value = " ".join(value.strip().split())
    return None if value.upper() in BAD_SUBJECTS else value


def catalog_and_semantics(bird_root: Path, semantic_path: Path) -> tuple[dict, dict]:
    catalog_rows = load_csv(bird_root / "artworks_dataset.csv")
    semantic_rows = load_csv(semantic_path)
    catalog = {r["image id"]: r for r in catalog_rows}
    semantic = {r["image_id"]: r for r in semantic_rows}
    if len(catalog) != len(catalog_rows) or len(semantic) != len(semantic_rows):
        raise ValueError("Duplicate artwork identifiers")
    if set(catalog) != set(semantic):
        raise ValueError(f"Catalogue/semantics ID mismatch: {set(catalog)^set(semantic)}")
    items = {}
    for poi, c in sorted(catalog.items()):
        s = semantic[poi]
        for a, b in [("room_id", "room_id"), ("painting name", "titre"), ("artist", "artiste")]:
            if c[a] != s[b]:
                raise ValueError(f"Metadata conflict {poi}: catalogue {a} vs semantic {b}")
        if s["statut"].strip().lower() != "ok":
            raise ValueError(f"Unreviewed semantic result for {poi}: {s['statut']}")
        main = subject(s["sujet_principal"])
        secondaries = []
        for part in s["sujets_secondaires"].split(";"):
            value = subject(part)
            if value and value != main and value not in secondaries:
                secondaries.append(value)
        for key in ["hook date", "unhook date"]:
            if c[key]:
                datetime.strptime(c[key], "%Y-%m-%d")
        items[poi] = {
            "poi_id": poi, "room_id": c["room_id"], "floor": int(c["floor"]),
            "hook_date": c["hook date"] or None,
            "unhook_date": c["unhook date"] or None,
            "subject_main": main, "subject_secondary": secondaries,
            "annotation_provenance": {
                "pipeline_status": s["statut"],
                "identity_control": s["controle_identite"],
                "subject_disagreement_with_previous": s["desaccord_sujet"].strip().lower() == "true",
                "source_page": s["source_page"] or None,
                "image_sha256": None,  # Source JSON archive retains the image checksum.
                "model": s["model"], "processed_at_utc": s["date_utc"],
            },
            "reserved_for_plm_not_model_features": {
                "title": c["painting name"], "artist": c["artist"],
                "visual_description": s["description_visuelle"],
                "style_raw": s["style_visuel_probable"],
            },
        }
    return items, {"catalogue": len(catalog), "semantic_rows": len(semantic),
                   "unknown_main": sum(x["subject_main"] is None for x in items.values()),
                   "unverified_ai_identity": sum(x["annotation_provenance"]["identity_control"] == "retenu_par_IA_non_verifie_humain" for x in items.values()),
                   "subject_disagreements": sum(x["annotation_provenance"]["subject_disagreement_with_previous"] for x in items.values())}


def parse_events(path: Path, items: dict, bad: list[dict]) -> list[dict]:
    rows = load_csv(path)
    output = []
    for source_row, r in enumerate(rows, start=2):
        try:
            t = float(r["timestamp"])
            if not math.isfinite(t) or t < 0:
                raise ValueError("nonfinite/negative time")
            poi = r["paintingId"]
            if poi not in items:
                raise ValueError("POI absent from catalogue")
            floor = int(r["floorNumber"])
            if floor != items[poi]["floor"]:
                raise ValueError("floor conflict")
            output.append({"t": t, "poi": poi, "source_row": source_row})
        except (ValueError, KeyError, TypeError) as e:
            bad.append({"file": path.name, "source_row": source_row, "raw": r, "reason": str(e)})
    output.sort(key=lambda e: (e["t"], e["source_row"]))
    return output


def pair_visit(starts: list[dict], ends: list[dict], visit_id: str, items: dict) -> tuple[list[dict], list[dict], dict]:
    """One-to-one pairing within [start_i, start_{i+1}], with explicit leftovers."""
    by_poi = defaultdict(list)
    for j, e in enumerate(ends):
        by_poi[e["poi"]].append((j, e))
    used = set()
    events, ambiguity = [], []
    for i, s in enumerate(starts):
        next_start = starts[i+1]["t"] if i+1 < len(starts) else None
        lower, upper = s["t"], float("inf") if next_start is None else next_start
        candidates = [(j, e) for j, e in by_poi[s["poi"]]
                      if j not in used and lower <= e["t"] <= upper + 1e-9]
        chosen = min(candidates, key=lambda x: (x[1]["t"], x[1]["source_row"])) if candidates else None
        if len(candidates) > 1:
            ambiguity.append({"visit": visit_id, "event_index": i, "end_rows": [e["source_row"] for _, e in candidates]})
        if chosen:
            used.add(chosen[0])
        end = chosen[1]["t"] if chosen else None
        events.append({"event_index": i, "poi_id": s["poi"],
                       "room_id": items[s["poi"]]["room_id"],
                       "t_start_seconds": lower, "start_source_row": s["source_row"],
                       "t_end_observed_seconds": end,
                       "end_source_row": chosen[1]["source_row"] if chosen else None,
                       "duration_observed_seconds": end-lower if end is not None else None,
                       "pair_status": "observed" if chosen else "missing_end",
                       "t_end_imputed_offline_seconds": None,
                       "duration_imputed_offline_seconds": None,
                       "duration_origin": "measured" if chosen else "unavailable"})
    measured = [e["duration_observed_seconds"] for e in events if e["pair_status"] == "observed" and e["duration_observed_seconds"] > 0]
    full_mean = sum(measured)/len(measured) if measured else None
    for i, e in enumerate(events):
        if e["pair_status"] == "observed" or i+1 == len(events) or full_mean is None:
            continue
        gap = events[i+1]["t_start_seconds"]-e["t_start_seconds"]
        if gap <= 0:
            continue
        duration = min(gap, (gap+full_mean)/2)
        e["duration_imputed_offline_seconds"] = duration
        e["t_end_imputed_offline_seconds"] = e["t_start_seconds"]+duration
        e["duration_origin"] = "imputed_offline_future_visit_mean_DO_NOT_USE_AS_FEATURE"
    leftovers = []
    for j,e in enumerate(ends):
        if j in used:
            continue
        earlier=[s for s in starts if s["poi"]==e["poi"] and s["t"]<=e["t"]]
        reason="no_prior_start_of_same_poi" if not earlier else "extra_or_outside_nonoverlapping_start_interval"
        leftovers.append({"visit":visit_id,"poi_id":e["poi"],"source_row":e["source_row"],
                          "timestamp":e["t"],"reason":reason})
    return events, leftovers, {"ambiguous_candidates": ambiguity, "full_visit_measured_mean_seconds": full_mean}


def prepare(bird_root: Path, semantic_csv: Path, output: Path) -> dict:
    bird_root, semantic_csv, output = Path(bird_root), Path(semantic_csv), Path(output)
    items, catalog_audit = catalog_and_semantics(bird_root, semantic_csv)
    starts = sorted((bird_root/"start_obs_artworks").glob("items_*.csv"))
    if not starts:
        raise ValueError("No BIRD start observations")
    visitors = {}
    bad_start, bad_end, leftovers, ambiguous, source_files = [], [], [], [], []
    for start_path in starts:
        name = start_path.stem
        match = re.fullmatch(r"items_(\d{12})", name)
        if not match:
            raise ValueError(f"Unexpected visit filename: {name}")
        visit_date = datetime.strptime(match.group(1), "%Y%m%d%H%M").date().isoformat()
        end_path = bird_root/"end_obs_artworks"/(name+"_end.csv")
        if not end_path.is_file():
            raise ValueError(f"Missing end log {end_path}")
        s = parse_events(start_path, items, bad_start)
        e = parse_events(end_path, items, bad_end)
        if not s:
            raise ValueError(f"Visit {name} without valid starts")
        source_files.extend([start_path, end_path])
        paired, unpaired_ends, diag = pair_visit(s, e, name, items)
        if any(a["t_start_seconds"] >= b["t_start_seconds"] for a,b in zip(paired,paired[1:])):
            raise ValueError(f"Non-strict start timestamps in visit {name}; resolve explicitly")
        visitors[name] = {"visit_id": name, "visit_date_from_filename": visit_date,
                          "visit_date_assumption": "Filename encodes local wall-clock; date only used for availability, not as learned calendar feature",
                          "events": paired, "measured_visit_mean_offline_seconds": diag["full_visit_measured_mean_seconds"]}
        leftovers.extend(unpaired_ends)
        ambiguous.extend(diag["ambiguous_candidates"])
    if bad_start:
        raise ValueError(f"Invalid BIRD start observations: {bad_start[:10]}; no start is silently discarded")
    sem = {"format": "stkg.cll.artworks.v1", "items": items,
           "model_feature_allowlist": ["poi_id", "room_id", "subject_main", "subject_secondary", "hook_date", "unhook_date"]}
    trajectories = {"format": VERSION, "time_unit": "seconds from visitor log origin",
                    "pairing_policy": "same POI, end in [start_i, next_start], earliest unused end; leftover ends audited",
                    "offline_imputation_policy": "min(next_start-start, ((next_start-start)+mean_of_reliable_full_visit_observed_durations)/2); NOT causally available",
                    "visitors": visitors}
    all_events = [x for v in visitors.values() for x in v["events"]]
    counts = Counter(x["pair_status"] for x in all_events)
    dates_conflicts = []
    for name, v in visitors.items():
        for e in v["events"]:
            a = items[e["poi_id"]]
            day = v["visit_date_from_filename"]
            if (a["hook_date"] and day < a["hook_date"]) or (a["unhook_date"] and day >= a["unhook_date"]):
                dates_conflicts.append((name, e["poi_id"], day))
    audit = {"format": "stkg.cll.audit.v1", "visits": len(visitors),
             "raw_starts": sum(len(load_csv(x)) for x in starts),
             "raw_ends": sum(len(load_csv(bird_root/"end_obs_artworks"/(x.stem+"_end.csv"))) for x in starts),
             "paired": counts["observed"], "unpaired_starts": counts["missing_end"],
             "offline_imputed": sum(x["duration_imputed_offline_seconds"] is not None for x in all_events),
             "offline_imputation_at_gap_cap": sum(x["duration_imputed_offline_seconds"] is not None and i+1<len(v["events"]) and abs(x["duration_imputed_offline_seconds"] - (v["events"][i+1]["t_start_seconds"]-x["t_start_seconds"])) < 1e-8 for v in visitors.values() for i,x in enumerate(v["events"])),
             "raw_invalid_starts": bad_start, "raw_invalid_ends": bad_end,
             "unpaired_valid_ends": leftovers, "ambiguous_end_candidates": ambiguous,
             "availability_conflicts": dates_conflicts,
             "availability_mask_allowed_without_catalogue_correction": not bool(dates_conflicts),
             "catalogue": catalog_audit,
             "distinct_observed_poi": len({x["poi_id"] for x in all_events}),
             "source_sha256": {str(x.relative_to(bird_root.parent)): sha256(x) for x in sorted(set(source_files+[bird_root/'artworks_dataset.csv',semantic_csv]))}}
    put_json(output/"trajectories_poi.json", trajectories)
    put_json(output/"artworks_semantics.json", sem)
    put_json(output/"data_audit.json", audit)
    return audit
