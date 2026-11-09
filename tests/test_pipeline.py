"""Nontrivial invariants: pairing, causality, train isolation, modality isolation, gradients."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from stkg_cll_poi.experiment import check_mask
from stkg_cll_poi.features import causal_duration,examples_by_split,graph_manifest,prefix_features,visitor_split
from stkg_cll_poi.model import CausalTransformerBaseline,STKGCLL,bidirectional_infonce,sequence_negative_mask
from stkg_cll_poi.preprocess import pair_visit

ROOT=Path(__file__).resolve().parent.parent

class ScientificIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traj=json.loads((ROOT/'data/processed/trajectories_poi.json').read_text())
        cls.art=json.loads((ROOT/'data/processed/artworks_semantics.json').read_text())
        cls.audit=json.loads((ROOT/'data/processed/data_audit.json').read_text())
        cls.split=visitor_split(cls.traj['visitors'],2022)
        cls.graph=graph_manifest(cls.traj,cls.art,cls.split)

    def test_coverage_and_pairing_audit(self):
        a=self.audit
        self.assertEqual((a['visits'],a['raw_starts'],a['raw_ends'],a['paired'],a['unpaired_starts']),(51,7204,5635,5631,1573))
        self.assertEqual(a['paired']+a['unpaired_starts'],a['raw_starts'])
        self.assertEqual(a['paired']+len(a['raw_invalid_ends'])+len(a['unpaired_valid_ends']),a['raw_ends'])
        self.assertEqual(len(self.art['items']),359)
        self.assertEqual(len(a['availability_conflicts']),1)
        self.assertEqual(a['offline_imputation_at_gap_cap'],1201)
        for v in self.traj['visitors'].values():
            used=[e['end_source_row'] for e in v['events'] if e['end_source_row'] is not None]
            self.assertEqual(len(used),len(set(used)))
            for e,n in zip(v['events'],v['events'][1:]):
                self.assertLess(e['t_start_seconds'],n['t_start_seconds'])
                if e['t_end_observed_seconds'] is not None:
                    self.assertLessEqual(e['t_end_observed_seconds'],n['t_start_seconds']+1e-8)
                if e['duration_imputed_offline_seconds'] is not None:
                    gap=n['t_start_seconds']-e['t_start_seconds']
                    self.assertLessEqual(e['duration_imputed_offline_seconds'],gap+1e-8)
                    m=v['measured_visit_mean_offline_seconds']
                    self.assertAlmostEqual(e['duration_imputed_offline_seconds'],min(gap,(gap+m)/2),places=7)

    def test_future_mutations_cannot_change_prefix_features(self):
        v=next(v for v in self.traj['visitors'].values() if len(v['events'])>20)
        e=copy.deepcopy(v['events']); current=9
        first=prefix_features(e,current,32)
        for j in range(current+1,len(e)):
            e[j]['poi_id']='FUTURE_CHANGED';e[j]['t_start_seconds']+=100000
            e[j]['duration_observed_seconds']=100000
            e[j]['t_end_observed_seconds']=e[j]['t_start_seconds']+100000
        e[current]['duration_imputed_offline_seconds']=100000
        e[current]['t_end_imputed_offline_seconds']=100000
        self.assertEqual(first,prefix_features(e,current,32))
        self.assertEqual(first[1][-1][1:4],[0.,0.,0.])

    def test_test_visitors_are_excluded_from_behavior_graph(self):
        changed=copy.deepcopy(self.traj)
        chosen=self.split['test'][0]
        for e in changed['visitors'][chosen]['events']:
            e['poi_id']='POI_FROM_HELDOUT_ONLY'
            e['t_start_seconds']+=100000
        self.assertEqual(self.graph,graph_manifest(changed,self.art,self.split))
        all_ids=set(self.split['train']+self.split['validation']+self.split['test'])
        self.assertEqual(len(all_ids),51)
        self.assertEqual([len(self.split[k]) for k in ('train','validation','test')],[35,8,8])

    def test_future_end_does_not_enter_causal_dwell(self):
        e=[{'poi_id':'a','t_start_seconds':0.,'t_end_observed_seconds':None,'duration_observed_seconds':None},
           {'poi_id':'b','t_start_seconds':10.,'t_end_observed_seconds':12.,'duration_observed_seconds':2.},
           {'poi_id':'c','t_start_seconds':30.,'t_end_observed_seconds':None,'duration_observed_seconds':None},
           {'poi_id':'d','t_start_seconds':50.,'t_end_observed_seconds':None,'duration_observed_seconds':None}]
        self.assertEqual(causal_duration(e,0,1),(0.,0,0))  # No earlier measured end yet.
        self.assertEqual(causal_duration(e,2,3),(11.,0,1)) # (20+2)/2, bounded by 20.
        self.assertEqual(causal_duration(e,3,3),(0.,0,0))
        e[3]['t_end_observed_seconds']=99999.;e[3]['duration_observed_seconds']=99949.
        self.assertEqual(causal_duration(e,2,3),(11.,0,1))

    def test_provenance_not_used_by_encoder_and_relations_receive_gradients(self):
        torch.manual_seed(12)
        a=STKGCLL(self.art['items'],self.graph,hidden=16,dropout=0,edge_dropout=0)
        changed=copy.deepcopy(self.art['items'])
        for v in changed.values():
            v['reserved_for_plm_not_model_features']['visual_description']='REPLACED'
            v['reserved_for_plm_not_model_features']['style_raw']='REPLACED'
            v['annotation_provenance']['model']='REPLACED'
        torch.manual_seed(12)
        b=STKGCLL(changed,self.graph,hidden=16,dropout=0,edge_dropout=0)
        a.eval();b.eval()
        idx={v:i for i,v in enumerate(self.graph['poi_ids'])}
        sample=examples_by_split(self.traj,self.split,3)['train'][:4]
        h=torch.tensor([[idx[p] for p in e['poi_history']]+[0]*(3-len(e['poi_history'])) for e in sample])
        time=torch.zeros(len(sample),3,5);lens=torch.tensor([len(e['poi_history']) for e in sample])
        with torch.no_grad():
            self.assertTrue(torch.allclose(a(h,time,lens)[0],b(h,time,lens)[0]))
        a.train();scores,_,_=a(h,time,lens)
        label=torch.tensor([idx[e['target']] for e in sample]);F.cross_entropy(scores,label).backward()
        self.assertGreater(float(a.graph.proj_room.weight.grad.abs().sum()),0)
        self.assertGreater(float(a.graph.proj_main.weight.grad.abs().sum()),0)
        self.assertGreater(float(a.graph.proj_secondary.weight.grad.abs().sum()),0)
        self.assertGreater(float(a.graph.proj_out.weight.grad.abs().sum()),0)
        self.assertGreater(float(a.graph.edge_time_gate.weight.grad.abs().sum()),0)

    def test_contrast_and_availability_checks(self):
        x=torch.tensor([[1,2],[1,2],[3,4],[3,4]])
        lengths=torch.tensor([2,2,2,2]);target=torch.tensor([10,11,10,12])
        mask=sequence_negative_mask(x,lengths,target)
        self.assertFalse(mask[0,1]) # Exact duplicate history.
        self.assertFalse(mask[0,2]) # Same next artwork.
        self.assertTrue(mask[0,3])
        h=torch.randn(4,8,requires_grad=True)
        loss=bidirectional_infonce(h,h+0.1,mask,.2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward();self.assertTrue(torch.isfinite(h.grad).all())
        with self.assertRaises(ValueError):check_mask(self.audit,True)

    def test_attention_baseline_ignores_padded_future(self):
        torch.manual_seed(44)
        model=CausalTransformerBaseline(40,16,8,0,True).eval()
        a=torch.tensor([[1,0,0]])
        b=torch.tensor([[1,32,39]])
        t=torch.zeros(1,3,5);length=torch.tensor([1])
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(a,t,length)[0],model(b,t,length)[0],atol=1e-6))

    def test_subject_permutation_preserves_global_annotation_counts(self):
        torch.manual_seed(3)
        a=STKGCLL(self.art['items'],self.graph,hidden=16,dropout=0,edge_dropout=0,variant='full')
        torch.manual_seed(3)
        b=STKGCLL(self.art['items'],self.graph,hidden=16,dropout=0,edge_dropout=0,variant='shuffled_subjects')
        self.assertTrue(torch.equal(torch.sort(a.graph.main_id).values,torch.sort(b.graph.main_id).values))
        self.assertTrue(torch.allclose(a.graph.secondary.sum(0),b.graph.secondary.sum(0),atol=1e-5))
        self.assertFalse(torch.equal(a.graph.main_id,b.graph.main_id))
        self.assertTrue(torch.equal(a.graph.room_id,b.graph.room_id))

if __name__=='__main__':unittest.main()
