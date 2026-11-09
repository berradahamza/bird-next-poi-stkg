"""Scientific invariants for topology, information flow and data isolation."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from stkg_cll_poi.experiment import fit_feature_normalizer, load_run_graph, seed_all
from stkg_cll_poi.features import examples_by_split, graph_manifest, visitor_split
from stkg_cll_poi.heterogeneous import HETERO_VARIANTS, build_heterogeneous_graph, graph_summary
from stkg_cll_poi.model import STKGCLL, Channels, RelationalGraph
from stkg_cll_poi.preprocess import put_json, sha256
from stkg_cll_poi.spatial import load_spatial, validate_spatial, room_nodes, room_adjacency_edges

ROOT = Path(__file__).resolve().parent.parent


class HeterogeneousTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_all(2022)
        cls.traj = json.loads((ROOT / "data/processed/trajectories_poi.json").read_text(encoding="utf-8"))
        cls.art = json.loads((ROOT / "data/processed/artworks_semantics.json").read_text(encoding="utf-8"))
        cls.split = visitor_split(cls.traj["visitors"], 2022)
        cls.behavior = graph_manifest(cls.traj, cls.art, cls.split)
        cls.raw = json.loads((ROOT / "data/raw/bird_adjacency.json").read_text(encoding="utf-8"))
        cls.spatial = load_spatial(ROOT / "data/raw/bird_adjacency.json", strict=True, verbose=False)
        cls.graph = build_heterogeneous_graph(cls.art["items"], cls.behavior, cls.spatial)

    def model(self, graph=None, seed=2022, edge_dropout=0):
        graph = self.graph if graph is None else graph
        torch.manual_seed(seed)
        return STKGCLL(self.art["items"], graph, hidden=16, dropout=0,
                       edge_dropout=edge_dropout, variant=graph["variant"])

    def test_spatial_topology_exactly_matches_input(self):
        r = self.spatial["validation"]
        self.assertEqual(r["issues"], [])
        expected = {(a, b) for a, neighbors in self.raw["neighbors"].items() for b in neighbors}
        edges, _ = room_adjacency_edges(self.spatial)
        self.assertEqual({(e["from"], e["to"]) for e in edges}, expected)
        self.assertEqual(len(edges), len(expected))
        self.assertTrue(all((b, a) in expected for a, b in expected))
        self.assertTrue(all(a != b for a, b in expected))
        self.assertEqual(r["cross_floor_edges"], [])
        self.assertEqual(len(self.graph["nodes"]["ROOM"]), len(self.raw["rooms"]))

    def test_invalid_topology_warns_and_never_invents_reverse(self):
        raw = copy.deepcopy(self.raw)
        a = next(r for r, ns in raw["neighbors"].items() if ns)
        b = raw["neighbors"][a][0]
        raw["neighbors"][b].remove(a)
        raw["neighbors"][a].extend([a, "UNKNOWN_ROOM", b])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clean = validate_spatial(raw)
        codes = clean["validation"]["issue_counts"]
        self.assertGreater(len(caught), 0)
        for code in ["asymmetric_neighbors", "invalid_room_reference", "self_loop", "duplicate_neighbor"]:
            self.assertIn(code, codes)
        self.assertNotIn(sorted([a, b]), [e["rooms"] for e in clean["adjacencies"]])
        with self.assertRaises(ValueError):
            validate_spatial(raw, strict=True, emit_warnings=False)

    def test_conflicting_duplicates_and_nonfinite_features_are_reported(self):
        raw = copy.deepcopy(self.raw)
        raw["adjacencies"].append(copy.deepcopy(raw["adjacencies"][0]))
        raw["adjacencies"][-1]["type"] = "conflict"
        room = next(iter(raw["rooms"]))
        raw["rooms"][room]["area_m2"] = float("nan")
        raw["rooms"][room].pop("floor")
        raw["adjacencies"][1]["midpoint"] = [float("inf"), 0]
        clean = validate_spatial(raw, emit_warnings=False)
        codes = clean["validation"]["issue_counts"]
        for code in ["conflicting_duplicate_adjacency", "invalid_area_m2", "invalid_floor", "missing_or_invalid_midpoint"]:
            self.assertIn(code, codes)
        self.assertTrue(any(e["type"] == "unknown" for e in clean["adjacencies"]))
        nodes, _ = room_nodes(clean)
        self.assertTrue(all(np.isfinite(n["area_normalized"]) for n in nodes))

    def test_static_normalization_is_per_floor_for_midpoints(self):
        edges, norms = room_adjacency_edges(self.spatial)
        for floor in norms["midpoint_by_floor"]:
            arr = np.array([e["midpoint_normalized"] for e in edges if str(e["floor"]) == floor])
            np.testing.assert_allclose(arr.mean(0), 0, atol=1e-12)
            np.testing.assert_allclose(arr.std(0), 1, atol=1e-12)
        nodes, _ = room_nodes(self.spatial)
        area = np.array([n["area_normalized"] for n in nodes])
        self.assertAlmostEqual(float(area.mean()), 0)
        self.assertAlmostEqual(float(area.std()), 1)

    def test_explicit_vertical_edge_preserved_without_cross_floor_geometry(self):
        raw = copy.deepcopy(self.raw)
        a = next(r for r, v in raw["rooms"].items() if v["floor"] == 0)
        b = next(r for r, v in raw["rooms"].items() if v["floor"] == 1)
        raw["neighbors"][a].append(b)
        raw["neighbors"][b].append(a)
        raw["adjacencies"].append({"rooms": [a, b], "type": "stair", "same_floor_rooms": False, "midpoint": [10, 20]})
        spatial = validate_spatial(raw, strict=True)
        edges, _ = room_adjacency_edges(spatial)
        stair = [e for e in edges if e["type"] == "stair"]
        self.assertEqual(len(stair), 2)
        self.assertTrue(all(not e["midpoint_available"] for e in stair))

    def test_validation_and_test_cannot_change_behavior_or_normalization(self):
        changed = copy.deepcopy(self.traj)
        for fold in ["validation", "test"]:
            for vid in self.split[fold]:
                for e in changed["visitors"][vid]["events"]:
                    e["poi_id"] = "HELDOUT_ONLY"
                    e["t_start_seconds"] *= 17
                    e["duration_observed_seconds"] = 999999
                    e["t_end_observed_seconds"] = 9999999
        behavior = graph_manifest(changed, self.art, self.split)
        self.assertEqual(behavior, self.behavior)
        self.assertEqual(build_heterogeneous_graph(self.art["items"], behavior, self.spatial), self.graph)
        original = examples_by_split(self.traj, self.split, 3)
        modified = examples_by_split(changed, self.split, 3)
        norm = fit_feature_normalizer(original["train"], self.split["train"])
        self.assertEqual(norm, fit_feature_normalizer(modified["train"], self.split["train"]))
        with self.assertRaises(AssertionError):
            fit_feature_normalizer(original["validation"], self.split["train"])
        overlapping = copy.deepcopy(self.split)
        overlapping["train"].append(overlapping["test"][0])
        with self.assertRaises(AssertionError):
            graph_manifest(self.traj, self.art, overlapping)

    def test_original_transition_values_are_unchanged(self):
        disk = json.loads((ROOT / "data/processed/graph_train.json").read_text(encoding="utf-8"))
        self.assertEqual(self.behavior, disk)
        self.assertEqual(self.graph["transition_edges"], disk["transition_edges"])

    def test_ablations_have_the_expected_relations_and_finite_gradients(self):
        for variant in HETERO_VARIANTS:
            with self.subTest(variant=variant):
                graph = build_heterogeneous_graph(self.art["items"], self.behavior, self.spatial, variant)
                names = {r["name"] for r in graph["relations"]}
                self.assertEqual("located_in" in names, graph["options"]["room"])
                self.assertEqual("has_main_subject" in names, graph["options"]["subject"])
                self.assertEqual(any(n.startswith("adjacent_") for n in names), graph["options"]["spatial"])
                model = self.model(graph)
                logits, _, nodes = model(torch.tensor([[0, 1], [2, 3]]), torch.zeros(2, 2, 5), torch.tensor([2, 2]))
                self.assertEqual(tuple(logits.shape), (2, len(self.art["items"])))
                loss = F.cross_entropy(logits, torch.tensor([4, 5]))
                loss.backward()
                self.assertTrue(torch.isfinite(nodes).all())
                gradients = [p.grad for p in model.graph.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(float(model.graph.edge_time_gate.weight.grad.abs().sum()), 0)
                if graph["options"]["room"]:
                    self.assertGreater(float(model.graph.room.weight.grad.abs().sum()), 0)
                    self.assertGreater(float(model.graph.floor.weight.grad.abs().sum()), 0)
                if graph["options"]["subject"]:
                    self.assertGreater(float(model.graph.subject.weight.grad.abs().sum()), 0)
                for i, relation in enumerate(model.graph.relations):
                    self.assertGreater(float(model.graph.layers[0].scales[str(i)].grad.abs().sum()), 0, relation.name)

    def test_metadata_is_retained_but_text_is_not_encoded(self):
        graph = copy.deepcopy(self.graph)
        for poi in graph["nodes"]["POI"]:
            self.assertIn("title", poi)
            self.assertIn("artist", poi)
            poi["title"] = "CHANGED TITLE"
            poi["artist"] = "CHANGED ARTIST"
        a, b = self.model().eval(), self.model(graph).eval()
        self.assertTrue(torch.equal(a.graph(), b.graph()))

    def test_metadata_ablation_and_optional_geometry(self):
        no_metadata = build_heterogeneous_graph(self.art["items"], self.behavior, self.spatial, "poi_room", room_features=False)
        changed = copy.deepcopy(no_metadata)
        for room in changed["nodes"]["ROOM"]:
            room["area_normalized"] += 100
            room["floor_category"] = "OTHER"
        self.assertTrue(torch.equal(self.model(no_metadata).eval().graph(), self.model(changed).eval().graph()))
        no_geometry = copy.deepcopy(self.graph)
        for relation in no_geometry["relations"]:
            for edge in relation["edges"]:
                if "midpoint_normalized" in edge:
                    edge["midpoint_normalized"] = [100, 200]
        self.assertTrue(torch.equal(self.model().eval().graph(), self.model(no_geometry).eval().graph()))
        graph = build_heterogeneous_graph(self.art["items"], self.behavior, self.spatial, midpoint=True)
        changed = copy.deepcopy(graph)
        relation = next(r for r in changed["relations"] if r["name"].startswith("adjacent_"))
        relation["edges"][0]["midpoint_normalized"] = [100, 200]
        self.assertFalse(torch.allclose(self.model(graph).eval().graph(), self.model(changed).eval().graph()))

    def test_spatial_type_changes_poi_embeddings(self):
        changed = copy.deepcopy(self.graph)
        rel = next(r for r in changed["relations"] if r["name"] == "adjacent_door")
        border = next(r for r in changed["relations"] if r["name"] == "adjacent_border")
        # Exchange attributes/edges while keeping parameter shapes identical.
        rel["edges"][0], border["edges"][0] = border["edges"][0], rel["edges"][0]
        a, b = self.model().eval(), self.model(changed).eval()
        # Distinct learned categorical relation maps; initialized equally by design.
        for model in [a, b]:
            index = next(i for i, r in enumerate(model.graph.relations) if r.name == "adjacent_door")
            with torch.no_grad():
                model.graph.layers[0].scales[str(index)].fill_(2)
        self.assertFalse(torch.allclose(a.graph(), b.graph()))

    def test_static_edges_survive_complete_behavior_dropout(self):
        a = self.model(edge_dropout=1).train()
        graph = copy.deepcopy(self.graph)
        graph["transition_edges"] = []
        graph["relations"][0]["edges"] = []
        b = self.model(graph, edge_dropout=0).train()
        self.assertTrue(torch.allclose(a.graph(), b.graph(), atol=1e-6))
        a.graph().square().sum().backward()
        self.assertGreater(float(a.graph.room.weight.grad.abs().sum()), 0)
        self.assertGreater(float(a.graph.subject.weight.grad.abs().sum()), 0)

    def test_two_hops_carry_peer_poi_information(self):
        for variant, field in [("poi_room", "room_id"), ("poi_subject", "subject_main")]:
            graph = build_heterogeneous_graph(self.art["items"], self.behavior, self.spatial, variant)
            graph["transition_edges"] = []
            graph["relations"][0]["edges"] = []
            ids = graph["poi_ids"]
            a, b = next((i, j) for i in ids for j in ids if i != j and self.art["items"][i][field]
                        and self.art["items"][i][field] == self.art["items"][j][field])
            model = self.model(graph).eval()
            before = model.graph().detach()
            with torch.no_grad():
                model.graph.item.weight[ids.index(a), 0] += 100
            after = model.graph().detach()
            self.assertFalse(torch.allclose(before[ids.index(b)], after[ids.index(b)]), variant)

    def test_poi_only_one_layer_reproduces_legacy_behavior_formula(self):
        graph = build_heterogeneous_graph(self.art["items"], self.behavior, self.spatial, "poi_only", layers=1)
        model = self.model(graph).graph.eval()
        legacy = RelationalGraph(self.art["items"], self.behavior, 16, 0, 0, Channels(True, False, False, False, True)).eval()
        layer = model.layers[0]
        legacy.item.load_state_dict(model.item.state_dict())
        legacy.proj_id.load_state_dict(layer.self_maps["POI"].state_dict())
        legacy.proj_out.load_state_dict(layer.proj_out.state_dict())
        legacy.proj_in.load_state_dict(layer.proj_in.state_dict())
        legacy.norm.load_state_dict(layer.norms["POI"].state_dict())
        legacy.edge_time_gate.load_state_dict(model.edge_time_gate.state_dict())
        self.assertTrue(torch.allclose(legacy(), model(), atol=1e-6))

    def test_frozen_run_rejects_tampering_and_heldout_provenance(self):
        with tempfile.TemporaryDirectory() as path:
            run = Path(path)
            put_json(run / "graph_train.json", self.graph)
            config = {"variant": "poi_room_subject", "run_input_sha256": {"graph_train.json": sha256(run / "graph_train.json")}}
            self.assertEqual(load_run_graph(run, config, self.behavior), self.graph)
            changed = copy.deepcopy(self.graph)
            changed["train_visitors"] = self.split["test"]
            put_json(run / "graph_train.json", changed)
            with self.assertRaises(ValueError):
                load_run_graph(run, config, self.behavior)
            config["run_input_sha256"]["graph_train.json"] = sha256(run / "graph_train.json")
            with self.assertRaises(AssertionError):
                load_run_graph(run, config, self.behavior)

    def test_empty_subjects_and_missing_room_are_explicit(self):
        items = copy.deepcopy(self.art["items"])
        for item in items.values():
            item["subject_main"] = None
            item["subject_secondary"] = []
        poi = next(iter(items))
        items[poi]["room_id"] = "MISSING"
        with warnings.catch_warnings(record=True) as caught:
            graph = build_heterogeneous_graph(items, self.behavior, self.spatial)
        self.assertTrue(caught)
        summary = graph_summary(graph)
        self.assertEqual(summary["poi_without_room"], [poi])
        self.assertEqual(summary["node_counts"]["SUBJECT"], 0)
        self.assertTrue(torch.isfinite(self.model(graph).graph()).all())

    def test_spatial_signal_cannot_be_silently_disabled_by_one_layer(self):
        with self.assertRaises(ValueError):
            build_heterogeneous_graph(self.art["items"], self.behavior, self.spatial, layers=1)
        with self.assertRaises(ValueError):
            build_heterogeneous_graph(self.art["items"], self.behavior, self.spatial, "poi_only", midpoint=True)


if __name__ == "__main__":
    unittest.main()
