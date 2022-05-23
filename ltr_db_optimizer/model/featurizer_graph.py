import os
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import ltr_db_optimizer.enumeration_algorithm.enumeration_node as nodes

# Global vectors used for training, for normalizing the vectors
min_vec = np.array([0.0, 0.0, 0.0, 1.0, 5.0, 5.0])
max_vec = np.array([1.0, 1.0, 7.0, 4.801740e+08, 6.001215e+06, 6.001215e+06])
tree_high = 9410970000.0
tree_low = 1.0

class FeatureExtractorGraph:
    # We used the following SQL parts:
    # Sort
    # Join: Merge Join, Nested Loop Join, Hash Join
    # Aggregate: Stream Aggregate, Hash Aggregate
    # Scan: Index Scan, Table Scan
    type_vector = ["sort", "stream_aggregate", "hash_aggregate", "merge_join", "nested_loop_join", "hash_join", "index_scan", "table_scan", "null"]
    
    def __init__(self, with_cost = True, small_version=False):
        self.vector_length = len(self.type_vector)
        self.with_cost = with_cost
        if with_cost:
            self.vector_length += 1#2 # for estimated subtree cost and estimated rows
        self.small_version = small_version
        self.rows = {}
        
    def featurize_node(self, node):
        """
        Featurize plan in form of a graph to a tree shaped form
        """
        none_child = np.zeros(self.vector_length)
        none_child[8] = 1
        
        if node.has_featurized_plan():
            return node.get_featurized_plan(), node.estimated_rows
        
        if node.name in self.type_vector:
            this_vector = np.zeros(self.vector_length)
            this_vector[self.type_vector.index(node.name)] = 1
        else:
            # If it is not in type vector, there should also be only one child
            return self.featurize_node(node.get_left_child())
        
        if self.with_cost:
            # insert cost here
            this_vector[-1] = node.estimated_rows # changed -2
            rows = node.estimated_rows
            
        # test length of children (2,1,0)
        if node.has_right_child():
            # called for joins
            left_child,_ = self.featurize_node(node.get_left_child())
            right_child,_ = self.featurize_node(node.get_right_child())
        elif node.has_left_child():
            # called for sorts and aggregates
            left_child,_ = self.featurize_node(node.get_left_child())
            right_child = (none_child)
        else:
            # called for scans
            node.set_featurized_plan((this_vector))
            return (this_vector), node.estimated_rows
        
        node.set_featurized_plan((this_vector, left_child, right_child))
        return (this_vector, left_child, right_child), node.estimated_rows
    
    def match_cost_plan(self, execution_plan, cost_plan):
        """
        For an execution plan (graph) and a cost plan (XML from SQL Server),
        insert the estimated number of rows into the plan.
        """
        
        cost_parts = cost_plan.split("<")
        parts_cost = []
        
        for part_num, part in enumerate(cost_parts):
            if part.startswith("RelOp"):
                sub_parts = part.split('"')
                sub_parts_cost = []
                for idx, sub in enumerate(sub_parts):
                    if "PhysicalOp" in sub:
                        sub_parts_cost.append(sub_parts[idx+1])
                    if "EstimateRows" in sub:
                        sub_parts_cost.append(float(sub_parts[idx+1]))
                parts_cost.append(tuple(sub_parts_cost))

        self.append_features(execution_plan, parts_cost)
    
    def append_features(self, execution_plan, label_parts):
        """
        Here, the estimated rows are inserted into the correct child as part of its dictionary.
        Mostly, this function deals with some SQL Server's logic.
        """
        
            if not(execution_plan.name == "top" and execution_plan.get_left_child().name == "sort"):
                if execution_plan.name == "compute_scalar" and not label_parts[0][0] == "Compute Scalar":
                    curr_part = label_parts[0]
                else:
                    curr_part = label_parts.pop(0)
                if not execution_plan.has_rows():
                    rows_calc = (curr_part[1]-tree_low)/(tree_high-tree_low)
                    execution_plan.set_estimated_rows(rows_calc)
                    self.rows[execution_plan.id] = rows_calc
            if execution_plan.name not in ["index_scan", "table_scan"]:
                for child in execution_plan.get_children():
                    self.append_features(child, label_parts) 
    
    def append_cost(self, plan):
        """
        The number of rows of the last node should be equal for every equivalent plan.
        To not insert every plan into SQL Server, we test here if the number of 
        rows for an equivalent plan has been calculated already. Return True if the plan needs
        to be inserted into SQL Server.
        """
        
        if plan.has_rows():
            return False
        if plan.id in self.rows and plan.name != "sort":
            plan.set_estimated_rows(self.rows[plan.id])
            toggle = False
            if plan.has_right_child():
                toggle = toggle or self.append_cost(plan.get_right_child())
            if plan.has_left_child():
                toggle = toggle or self.append_cost(plan.get_left_child())
            return toggle
        elif plan.name == "sort":
            if plan.left_child.has_rows():
                plan.set_estimated_rows(plan.left_child.estimated_rows)
                return False
            else:
                return_val = self.append_cost(plan.get_left_child())
                if return_val:
                    return True
                plan.set_estimated_rows(plan.left_child.estimated_rows)
                return False
        return True


def get_left_child(node):
    if len(node) != 3:
        return None
    return node[1]

def get_right_child(node):
    if len(node) != 3:
        return None
    return node[2]
    
def get_features(node):
    if len(node) != 3:
        return node
    return node[0]
    