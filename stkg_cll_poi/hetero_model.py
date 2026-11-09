"""Compact heterogeneous message passing in plain PyTorch (no PyG required)."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def mean_messages(values, destination, count):
    total = values.new_zeros((count, values.shape[-1])).index_add_(0, destination, values)
    degree = values.new_zeros(count).index_add_(0, destination, values.new_ones(len(destination)))
    return total / degree.clamp_min(1).unsqueeze(1)


class TypedEdges(nn.Module):
    """Explicit edges, including independent main/secondary inverse relations."""
    def __init__(self, relation, indexes, midpoint):
        super().__init__()
        self.name = relation["name"]
        self.source_type, self.target_type = relation["source_type"], relation["target_type"]
        edges = relation["edges"]
        self.register_buffer("src", torch.tensor([indexes[self.source_type][e["from"]] for e in edges], dtype=torch.long))
        self.register_buffer("dst", torch.tensor([indexes[self.target_type][e["to"]] for e in edges], dtype=torch.long))
        self.spatial = self.name.startswith("adjacent_")
        if self.spatial:
            attrs = []
            for edge in edges:
                same = edge["same_floor_rooms"]
                row = [float(same is True), float(same is not None)]
                if midpoint:
                    row += [*edge["midpoint_normalized"], float(edge["midpoint_available"])]
                attrs.append(row)
            self.register_buffer("attrs", torch.tensor(attrs, dtype=torch.float32).reshape(-1, 5 if midpoint else 2))


class RelationalMeanLayer(nn.Module):
    """A shared full projection per source type and a diagonal map per relation.

    This avoids a full hidden x hidden matrix for every static relation.
    Behavioral incoming/outgoing maps retain the original weighted mean form.
    Updates are synchronous: layer l reads only states from layer l-1.
    """
    def __init__(self, types, relations, hidden, dropout, midpoint, final=False):
        super().__init__()
        # Final ROOM/SUBJECT updates cannot affect the POI output: omit their
        # unused parameters while retaining them as sources for the last layer.
        if final:
            types = ["POI"]
        self.relation_indexes = [i for i, r in enumerate(relations) if r.target_type in types]
        active = [relations[i] for i in self.relation_indexes]
        self.self_maps = nn.ModuleDict({t: nn.Linear(hidden, hidden, bias=False) for t in types})
        self.source_maps = nn.ModuleDict({t: nn.Linear(hidden, hidden, bias=False)
                                         for t in sorted({r.source_type for r in active})})
        self.scales = nn.ParameterDict({str(i): nn.Parameter(torch.ones(hidden)) for i in self.relation_indexes})
        self.proj_out = nn.Linear(hidden, hidden, bias=False)
        self.proj_in = nn.Linear(hidden, hidden, bias=False)
        self.norms = nn.ModuleDict({t: nn.LayerNorm(hidden) for t in types})
        self.spatial_features = nn.Linear(5 if midpoint else 2, hidden, bias=False) if any(r.spatial for r in active) else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, relations, src, dst, weights):
        z = {t: self.self_maps[t](nodes[t]) for t in self.self_maps}
        projected = {t: self.source_maps[t](nodes[t]) for t in self.source_maps}
        for index in self.relation_indexes:
            relation, scale = relations[index], self.scales[str(index)]
            messages = projected[relation.source_type][relation.src] * scale
            if relation.spatial:
                messages = messages + self.spatial_features(relation.attrs)
            target = relation.target_type
            z[target] = z[target] + mean_messages(messages, relation.dst, len(nodes[target]))
        base = nodes["POI"]
        if src.numel():
            count = len(base)
            denom_out = base.new_zeros(count).index_add_(0, src, weights).clamp_min(1e-9)
            denom_in = base.new_zeros(count).index_add_(0, dst, weights).clamp_min(1e-9)
            outgoing = torch.zeros_like(base).index_add_(0, src, base[dst] * weights.unsqueeze(1)) / denom_out.unsqueeze(1)
            incoming = torch.zeros_like(base).index_add_(0, dst, base[src] * weights.unsqueeze(1)) / denom_in.unsqueeze(1)
            z["POI"] = z["POI"] + self.proj_out(outgoing) + self.proj_in(incoming)
        return {t: self.dropout(F.gelu(self.norms[t](value))) for t, value in z.items()}


class HeterogeneousGraph(nn.Module):
    def __init__(self, graph, hidden, dropout, edge_dropout):
        super().__init__()
        options = graph["options"]
        self.item_index = {p: i for i, p in enumerate(graph["poi_ids"])}
        self.item = nn.Embedding(len(self.item_index), hidden)
        self.edge_dropout = edge_dropout
        self.dropout = nn.Dropout(dropout)  # Used by the unchanged sequence encoder.
        self.use_room_features = options["room_features"]
        self.has_room = "ROOM" in graph["nodes"]
        self.has_subject = "SUBJECT" in graph["nodes"]
        indexes = {t: {v["id"]: i for i, v in enumerate(nodes)} for t, nodes in graph["nodes"].items()}
        if self.has_room:
            rooms = graph["nodes"]["ROOM"]
            self.room = nn.Embedding(len(rooms), hidden)
            if self.use_room_features:
                floors = {f: i for i, f in enumerate(sorted({n["floor_category"] for n in rooms}))}
                self.floor = nn.Embedding(len(floors), hidden)
                self.area_projection = nn.Linear(2, hidden, bias=False)
                self.register_buffer("room_floor", torch.tensor([floors[n["floor_category"]] for n in rooms], dtype=torch.long))
                self.register_buffer("room_area", torch.tensor([[n["area_normalized"], float(n["area_available"])] for n in rooms], dtype=torch.float32))
        if self.has_subject:
            self.subject = nn.Embedding(len(indexes["SUBJECT"]), hidden)
        self.relations = nn.ModuleList([TypedEdges(r, indexes, options["midpoint"])
                                       for r in graph["relations"] if r["name"] != "next_poi"])
        edges = graph["transition_edges"]
        self.register_buffer("src", torch.tensor([self.item_index[e["from"]] for e in edges], dtype=torch.long))
        self.register_buffer("dst", torch.tensor([self.item_index[e["to"]] for e in edges], dtype=torch.long))
        self.register_buffer("edge_weight", torch.tensor([math.log1p(e["count"]) for e in edges], dtype=torch.float32))
        norm = graph["normalization"]["behavior"]
        if norm["fitted_from"] != "train_transition_edges_only" or norm["train_visitors"] != graph["train_visitors"]:
            raise AssertionError("Temporal edge normalization must use this run's train visitors")
        features = torch.tensor([[e[k] for k in norm["features"]] for e in edges], dtype=torch.float32).reshape(-1, 3)
        features = (features - torch.tensor(norm["mean"])) / torch.tensor(norm["std"])
        self.register_buffer("edge_time", features)
        self.edge_time_gate = nn.Linear(3, 1)
        nn.init.zeros_(self.edge_time_gate.weight)
        nn.init.zeros_(self.edge_time_gate.bias)
        self.layers = nn.ModuleList([RelationalMeanLayer(list(graph["nodes"]), self.relations, hidden, dropout, options["midpoint"],
                                                       final=i == options["layers"]-1)
                                    for i in range(options["layers"])])

    def forward(self):
        nodes = {"POI": self.item.weight}
        if self.has_room:
            nodes["ROOM"] = self.room.weight
            if self.use_room_features:
                nodes["ROOM"] = nodes["ROOM"] + self.floor(self.room_floor) + self.area_projection(self.room_area)
        if self.has_subject:
            nodes["SUBJECT"] = self.subject.weight
        weights = self.edge_weight * 2 * torch.sigmoid(self.edge_time_gate(self.edge_time).squeeze(-1))
        if self.training and self.edge_dropout:
            # One mask per contrastive view, shared by the two GNN layers.
            # All static/inverse/adjacency edges remain intact.
            weights = weights * (torch.rand_like(weights) >= self.edge_dropout).to(weights.dtype)
        for layer in self.layers:
            nodes = layer(nodes, self.relations, self.src, self.dst, weights)
        return nodes["POI"]
