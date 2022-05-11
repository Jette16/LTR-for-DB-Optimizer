import networkx as nx
import random
import torch

import ltr_db_optimizer.enumeration_algorithm.utils as utils 
import ltr_db_optimizer.enumeration_algorithm.enumeration_node as nodes
from ltr_db_optimizer.model.model_structures.comparison_net2 import LTRComparisonNet
from ltr_db_optimizer.model.model_structures.comparison_model import LTRComparisonNet as LTRComparisonNet_comp

class DPccp:
    
    def __init__(self, model = None, graph = None, joiner = None, top_k = 1, comparison = False):
        if model:
            if not comparison:
                self.model = LTRComparisonNet(10,6)
                self.model.load_state_dict(torch.load(model))
            else:
                self.model = LTRComparisonNet_comp(10,6)
                self.model.load_state_dict(torch.load(model))                
        else:
            self.model = None
        self.graph = graph
        self.joiner = joiner
        self.top_k = top_k
    
    def enumerate(self, name_in_data=False):
        """
        :param graph: networkx Graph in BFS
        """
        best_parts = {}
        full = ""
        
        for i in list(self.graph.nodes):
            dict_name = str(i)
            i_name = dict_name if not name_in_data else nx.get_node_attributes(self.graph, "old_name")[i]
            best_parts[dict_name] = self.joiner.get_scan(i_name)
        
        for csg, cmp in self.get_csg_cmp_pairs():
            csg_name = self.to_name(csg)
            cmp_name = self.to_name(cmp)
            full_name = self.to_name(csg, cmp)
            assert csg_name in best_parts
            assert cmp_name in best_parts
            
            # wenn nicht länge 1
            # filtern für beste gefundene Lösung von csg und cmp
            # alle möglichen Varianten (verschiedene Join Types und cmp-csg vs. csg-cmp) einfügen in best_parts[full_name]
            if csg_name in best_parts:
                if len(best_parts[csg_name]) > self.top_k:
                    best_parts[csg_name] = self.reduce(best_parts[csg_name])
                right = best_parts[csg_name]
                
            if cmp_name in best_parts:
                if len(best_parts[cmp_name]) > self.top_k:
                    best_parts[cmp_name] = self.reduce(best_parts[cmp_name])
                
                left = best_parts[cmp_name]
                
            possible_joins = self.joiner.get_join_possibilities(right, left)
            
            if full_name not in best_parts.keys():
                best_parts[full_name] = possible_joins
            else:
                best_parts[full_name].extend(possible_joins)
                
            full = full_name
        # gib besten Subplan zurück
        if len(best_parts[full]) > self.top_k:
            return self.reduce(best_parts[full])#[0]
        else:
            return best_parts[full]
    
    def reduce(self, plans, last = False):
        # hier noch model einfügen
        if self.model is None:
            return random.sample(plans, self.top_k)
        else:
            # should be the same sql for all
            prepared_plans = self.prepare_plans(plans, last)
            assert len(prepared_plans) == 2
            if len(prepared_plans[1]) == 1:
                return prepared_plans[1]
            predictions = self.model.predict_all(*prepared_plans).t()
            # get top_k plans
            k = self.top_k if not last else 1
            return [plans[i] for i in torch.topk(predictions, k, dim = -1).indices[0]]
                
    
    def prepare_plans(self, plans):
        return plans
        
    def get_csg_cmp_pairs(self):
        for csg in self.enumerate_csg():
            for cmp in self.enumerate_cmp(csg):
                yield csg, cmp
    
    def enumerate_csg(self):
        # For all nodes i in reversed BFS
        for i in reversed(list(self.graph.nodes)):
            yield from self.yield_enumerate_csg(i)
    
    def yield_enumerate_csg(self, i: int):
        def filter_smaller(n):
            return n <= i
        yield [i]
        yield from self.enumerate_csg_rec([i], nx.subgraph_view(self.graph, filter_node=filter_smaller).nodes)
        
    def get_union_nodes(self, S, subset):
        try:
            subset = list(subset)
            S = list(S)
        except:
            raise Exception("subset should be convertible to list")
            
        def filter_with_subset(n):
            return n in S or n in subset
        
        return nx.subgraph_view(self.graph, filter_node=filter_with_subset).nodes
        
    def enumerate_csg_rec(self, S, X):
        N = self.get_neighbors_of_subgraph(S, X)
        for subset in utils.powerset(N):
            yield self.get_union_nodes(S, subset)
            
        for subset in utils.powerset(N):
            yield from self.enumerate_csg_rec(self.get_union_nodes(S, subset), 
                                              self.get_union_nodes(X, N))
        
    def get_neighbors_of_subgraph(self, S, X):
        assert all([node in X for node in S])
        N = []
        for node in S:
            N.extend(self.graph.neighbors(node)-X)
        return set(N)
        
    def enumerate_cmp(self, S_1):
        X = self.get_union_nodes(S_1, self.get_b_min(S_1))
        N = self.get_neighbors_of_subgraph(S_1, X)
        for node in reversed(list(N)):
            yield [node]
            yield from self.enumerate_csg_rec([node], self.get_union_nodes(X, N))
                           
    def get_b_min(self, S_1: list):
        minimum = min(S_1)
        def filter_higher(n):
            return n <= minimum
        return nx.subgraph_view(self.graph, filter_node=filter_higher).nodes
        
    def to_name(self, nodes_1, nodes_2 = None):
        nodes_1 = list(nodes_1)
        if nodes_2:
            nodes_1 += list(nodes_2)
        return "".join(str(i) for i in sorted(nodes_1))
    
            
        
            
        
                           