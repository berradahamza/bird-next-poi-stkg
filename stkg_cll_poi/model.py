
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .heterogeneous import HETERO_VARIANTS, ablation_options
from .hetero_model import HeterogeneousGraph


@dataclass(frozen=True)
class Channels:
    behavior: bool = True
    room: bool = True
    main: bool = True
    secondary: bool = True
    time: bool = True


def channel_variant(name: str) -> Channels:
    if name in HETERO_VARIANTS:
        options = ablation_options(name)
        return Channels(True, options["room"], options["subject"], options["subject"], True)
    variants = {
        "full": Channels(), "poi_id": Channels(False,False,False,False,False),
        "transitions": Channels(True,False,False,False,False),
        "no_subject": Channels(True,True,False,False,True),
        "main_only": Channels(True,True,True,False,True),
        "no_time": Channels(True,True,True,True,False),
        "no_behavior": Channels(False,True,True,True,True),
        "no_room": Channels(True,False,True,True,True),
        "shuffled_subjects": Channels(),
    }
    if name not in variants:
        raise ValueError(f"Unknown variant {name}; choose {sorted(variants)}")
    return variants[name]


class RelationalGraph(nn.Module):
    """One relational message-passing layer with separately learned relation maps.

    Static typed neighbors are Room / MainSubject / SecondarySubject.
    Directed, weighted POI transitions are trained from train visitors only.
    No feature is constructed from artist, title, style, or descriptions.
    """
    def __init__(self, artwork_items: dict, graph: dict, hidden: int, dropout: float,
                 edge_dropout: float, channels: Channels,shuffle_subjects: bool = False):
        super().__init__()
        ids = graph["poi_ids"]
        rooms = {r:i for i,r in enumerate(graph["room_ids"])}
        subjects = {s:i for i,s in enumerate(graph["subject_ids"])}
        n, k = len(ids), len(subjects)
        item_index={s:i for i,s in enumerate(ids)}
        self.item_index=item_index
        self.n=n
        self.channels=channels
        self.edge_dropout=edge_dropout
        self.item=nn.Embedding(n,hidden)
        self.room=nn.Embedding(len(rooms),hidden)
        self.subject=nn.Embedding(k,hidden)
        self.proj_id=nn.Linear(hidden,hidden,bias=False)
        self.proj_room=nn.Linear(hidden,hidden,bias=False)
        self.proj_main=nn.Linear(hidden,hidden,bias=False)
        self.proj_secondary=nn.Linear(hidden,hidden,bias=False)
        self.proj_out=nn.Linear(hidden,hidden,bias=False)
        self.proj_in=nn.Linear(hidden,hidden,bias=False)
        self.edge_time_gate=nn.Linear(3,1)
        nn.init.zeros_(self.edge_time_gate.weight)
        nn.init.zeros_(self.edge_time_gate.bias)
        self.norm=nn.LayerNorm(hidden)
        self.dropout=nn.Dropout(dropout)
        room_id=[]; main_id=[]; main_mask=[]; secondary=torch.zeros(n,k)
        semantic_sources=list(ids)
        if shuffle_subjects:
            random.Random(31415).shuffle(semantic_sources)
        for i,p in enumerate(ids):
            art=artwork_items[p]
            room_id.append(rooms[art["room_id"]])
            subjects_from=artwork_items[semantic_sources[i]]
            main=subjects_from["subject_main"]
            main_id.append(subjects[main] if main is not None else 0)
            main_mask.append(float(main is not None))
            for s in subjects_from["subject_secondary"]:
                secondary[i,subjects[s]] += 1
        secondary /= secondary.sum(1,keepdim=True).clamp_min(1)
        self.register_buffer("room_id",torch.tensor(room_id,dtype=torch.long))
        self.register_buffer("main_id",torch.tensor(main_id,dtype=torch.long))
        self.register_buffer("main_mask",torch.tensor(main_mask).unsqueeze(1))
        self.register_buffer("secondary",secondary)
        edge=graph["transition_edges"]
        self.register_buffer("src",torch.tensor([item_index[e["from"]] for e in edge],dtype=torch.long))
        self.register_buffer("dst",torch.tensor([item_index[e["to"]] for e in edge],dtype=torch.long))
        self.register_buffer("edge_weight",torch.tensor([math.log1p(e["count"]) for e in edge]))
        edge_time=torch.tensor([[e["mean_log_start_gap"],e["mean_log_causal_dwell"],e["causal_dwell_available_rate"]] for e in edge])
        if len(edge):
            mean=edge_time.mean(0);std=edge_time.std(0,unbiased=False).clamp_min(1e-6)
            edge_time=(edge_time-mean)/std
        self.register_buffer("edge_time",edge_time)

    def forward(self) -> torch.Tensor:
        base=self.item.weight
        z=self.proj_id(base)
        if self.channels.room:
            z=z+self.proj_room(self.room(self.room_id))
        if self.channels.main:
            z=z+self.proj_main(self.subject(self.main_id))*self.main_mask
        if self.channels.secondary:
            z=z+self.proj_secondary(self.secondary @ self.subject.weight)
        if self.channels.behavior and self.src.numel():
            weights=self.edge_weight
            if self.channels.time:
                weights=weights*2*torch.sigmoid(self.edge_time_gate(self.edge_time).squeeze(-1))
            if self.training and self.edge_dropout:
                weights=weights*(torch.rand_like(weights)>=self.edge_dropout).to(weights.dtype)
            denom_out=torch.zeros(self.n,device=base.device).index_add_(0,self.src,weights).clamp_min(1e-9)
            denom_in=torch.zeros(self.n,device=base.device).index_add_(0,self.dst,weights).clamp_min(1e-9)
            out=torch.zeros_like(base).index_add_(0,self.src,base[self.dst]*weights.unsqueeze(1))/denom_out.unsqueeze(1)
            incoming=torch.zeros_like(base).index_add_(0,self.dst,base[self.src]*weights.unsqueeze(1))/denom_in.unsqueeze(1)
            z=z+self.proj_out(out)+self.proj_in(incoming)
        return self.dropout(F.gelu(self.norm(z)))


class STKGCLL(nn.Module):
    def __init__(self, artwork_items: dict, graph: dict, hidden: int = 64, dropout: float = .2,
                 edge_dropout: float = .1, variant: str = "full"):
        super().__init__()
        self.channels=channel_variant(variant)
        if variant in HETERO_VARIANTS:
            if graph.get("format") != "stkg.cll.heterogeneous.v1" or graph["variant"] != variant:
                raise ValueError("The heterogeneous encoder requires its matching typed graph")
            self.graph=HeterogeneousGraph(graph,hidden,dropout,edge_dropout)
        else:
            # Historical graph encoder retained; the linear head requires a new checkpoint.
            self.graph=RelationalGraph(artwork_items,graph,hidden,dropout,edge_dropout,self.channels,
                                       shuffle_subjects=variant=="shuffled_subjects")
        self.temporal=nn.Sequential(nn.Linear(5,hidden),nn.Tanh())
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.head=nn.Linear(hidden,len(graph["poi_ids"]))
        self.seq_norm=nn.LayerNorm(hidden)
        self.seq_project=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.graph_project=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden))

    def forward(self, histories: torch.Tensor, temporal: torch.Tensor, lengths: torch.Tensor):
        nodes=self.graph()
        x=nodes[histories]
        if self.channels.time:
            x=x+self.temporal(temporal)
        x=F.dropout(x,p=self.graph.dropout.p,training=self.training)
        sequence,_=self.gru(x)
        b=torch.arange(len(lengths),device=lengths.device)
        h=self.seq_norm(sequence[b,lengths-1])
        logits=self.head(h)
        return logits,h,nodes


class CausalTransformerBaseline(nn.Module):
    """SASRec-style full-softmax baseline; optional observed-prefix temporal features.

    This is deliberately labelled "style": it is not a byte-for-byte port of
    SASRec or TiSASRec and never uses graph or semantic information.
    """
    def __init__(self, n_items:int,hidden:int,max_history:int,dropout:float,time_aware:bool):
        super().__init__()
        if hidden%4:
            raise ValueError("Transformer hidden dimension must be divisible by 4")
        self.time_aware=time_aware
        self.item=nn.Embedding(n_items,hidden)
        self.position=nn.Embedding(max_history,hidden)
        self.time_proj=nn.Linear(5,hidden) if time_aware else None
        layer=nn.TransformerEncoderLayer(d_model=hidden,nhead=4,dim_feedforward=hidden*2,
                                         dropout=dropout,batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(hidden)
        self.query=nn.Linear(hidden,hidden,bias=False)
        self.seq_project=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.logit_scale=nn.Parameter(torch.tensor(math.log(10.)))

    def forward(self,histories:torch.Tensor,temporal:torch.Tensor,lengths:torch.Tensor):
        batch,width=histories.shape
        positions=torch.arange(width,device=histories.device)
        x=self.item(histories)+self.position(positions)[None,:,:]
        if self.time_aware:
            x=x+self.time_proj(temporal)
        causal=torch.triu(torch.ones(width,width,dtype=torch.bool,device=histories.device),1)
        padding=positions[None,:]>=lengths[:,None]
        h=self.encoder(x,mask=causal,src_key_padding_mask=padding)
        vector=self.norm(h[torch.arange(batch,device=histories.device),lengths-1])
        nodes=self.item.weight
        score=(F.normalize(self.query(vector),dim=-1) @ F.normalize(nodes,dim=-1).T)*self.logit_scale.clamp(math.log(1.),math.log(50.)).exp()
        return score,vector,nodes


def bidirectional_infonce(first: torch.Tensor, second: torch.Tensor,
                          valid_negatives: torch.Tensor, temperature: float) -> torch.Tensor:
    """B x B cross-view InfoNCE, NOT the 2B x 2B SimCLR NT-Xent objective."""
    if temperature <= 0:
        raise ValueError("Positive temperature required")
    b=first.shape[0]
    if b < 2 or not valid_negatives.any():
        return first.sum()*0
    logits=F.normalize(first,dim=-1) @ F.normalize(second,dim=-1).T / temperature
    allowed=valid_negatives | torch.eye(b,dtype=torch.bool,device=first.device)
    logits=logits.masked_fill(~allowed,-torch.finfo(logits.dtype).max)
    eligible=valid_negatives.any(1)
    if not eligible.any():
        return first.sum()*0
    labels=torch.arange(b,device=first.device)
    forward=F.cross_entropy(logits[eligible],labels[eligible])
    reverse=F.cross_entropy(logits.T[eligible],labels[eligible])
    return (forward+reverse)/2


def sequence_negative_mask(histories: torch.Tensor, lengths: torch.Tensor,
                           targets: torch.Tensor, mask_same_target: bool = True) -> torch.Tensor:
    b=histories.size(0)
    different_history=~((histories[:,None,:]==histories[None,:,:]).all(-1) & (lengths[:,None]==lengths[None,:]))
    valid=different_history & ~torch.eye(b,dtype=torch.bool,device=histories.device)
    if mask_same_target:
        valid=valid & (targets[:,None]!=targets[None,:])
    return valid


def node_view_loss(model: STKGCLL, nodes_a: torch.Tensor, nodes_b: torch.Tensor,
                   histories: torch.Tensor, lengths: torch.Tensor, temperature: float) -> torch.Tensor:
    valid_positions=torch.arange(histories.shape[1],device=histories.device)[None,:]<lengths[:,None]
    chosen=torch.unique(histories[valid_positions])
    if len(chosen)<2:
        return nodes_a.sum()*0
    valid=~torch.eye(len(chosen),dtype=torch.bool,device=chosen.device)
    return bidirectional_infonce(model.graph_project(nodes_a[chosen]),
                                 model.graph_project(nodes_b[chosen]),valid,temperature)
