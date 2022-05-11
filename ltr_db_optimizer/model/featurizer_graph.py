import os
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import ltr_db_optimizer.enumeration_algorithm.enumeration_node as nodes

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
            left_child,_ = self.featurize_node(node.get_left_child())
            right_child,_ = self.featurize_node(node.get_right_child())
        elif node.has_left_child():
            left_child,_ = self.featurize_node(node.get_left_child())
            right_child = (none_child)
        else:
            node.set_featurized_plan((this_vector))
            return (this_vector), node.estimated_rows
        
        node.set_featurized_plan((this_vector, left_child, right_child))
        return (this_vector, left_child, right_child), node.estimated_rows
    
    def match_cost_plan(self, execution_plan, cost_plan):
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
            #full_execution_plan = {"operator": execution_plan[0], "children": []}
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
    
    
def get_features_with_cost_from_folder(plans_folder, cost_folder, return_featurized=True):
    feature_ext = FeatureExtractor()
    
    featurized_trees = {}
    featurized_vecs = {}
    
    for file in os.listdir(cost_folder):
        if file.endswith(".txt"):
            file_name = file.split(".")[0]
            job_nr = file.split("_")[0]
            version_nr = file_name.split("_")[1]
            
            cost_plan = ""
            with open(cost_folder+"/"+file, "r") as f:
                for line in f:
                    cost_plan += line
            try:
                with open(plans_folder+"/"+job_nr+"/"+version_nr+".pickle", "rb") as d:
                    execution_plan = pickle.load(d)
            except:
                #print(f"Problems finding plan for {file}")
                continue
            try:
                full_execution_plan = feature_ext.match_cost_plan(execution_plan, cost_plan)
            except:
                print(file)
                continue
            if return_featurized:
                featurized_plan, rows = feature_ext.featurize_plan(full_execution_plan)
                query_vec = feature_ext.featurize_query(job_nr, rows)
                featurized_trees[file_name] = featurized_plan
                featurized_vecs[file_name] = query_vec
            else:
                featurized_trees[file_name] = full_execution_plan
    return featurized_vecs, featurized_trees

################## Not needed for testing

def featurize_with_labels(plans_folder, cost_folder, label_csv, max_score = 50, score_function = "special", extra_for_min = True, special_border = 0.95):
    featurized_vecs, featurized_trees = get_features_with_cost_from_folder(plans_folder, cost_folder)
    label_dict = {}
    
    df = pd.read_csv(label_csv, index_col = 0)
    df["Job_nr"] = df["Unnamed: 0.1"].apply(lambda x: x.split("_")[0])
    
    times = []
    
    for job in pd.unique(df["Job_nr"]):
        temp_df = df[df["Job_nr"]==job].copy()
        a = np.array(temp_df["Sum"])
        if score_function == "special":
            temp = df[df["Job_nr"] == job]
            labels = temp.index
            x = np.array(temp["CPU time"])
            if len(x) == 0:
                continue
            #x[x<0] = np.max(x) * 2
            if np.min(x[x>=0]) != 0:
                x[x>=0] = x[x>=0]/np.min(x[x>=0])
            else:
                x = x + 1
            times.extend(list(zip(labels,x)))
        else:
            if score_function == "linear":
                a[a==-2] = max(a)*2 # Necessary, otherwise min(a) == -2
                temp_df["scores"] = calculate_linear_scores(a, n = max_score)
            elif score_function == "histogram":
                temp_df["scores"] = calculate_histogram_score(a, nr_bins = max_score, extra_bin_for_min = extra_for_min)
            elif score_function == "agglomerative":
                temp_df["scores"] = calculate_agglomerative_score(a, nr_clusters = max_score, extra_bin_for_min = extra_for_min)
            
            for idx, row in temp_df.iterrows():
                label_dict[idx] = row["scores"]
    if score_function == "special":
        labels, scores = calculate_special_score(times, max_score, special_border)
        for idx, s in enumerate(scores):
            label_dict[labels[idx]] = s
    
    return featurized_vecs, featurized_trees, label_dict

def calculate_special_score(scores, n, border_value):
    s = np.array([s[1] for s in scores])
    border = np.quantile(s[s >= 0], border_value)
    print(border)
    times = []
    for s in scores:
        if s[1] > border or s[1] < 0:
            times.append(border)
        else:
            times.append(s[1])
    times = np.array(times)
    labels = AgglomerativeClustering(n_clusters=n+1).fit_predict(times.reshape(-1,1))
    #labels = clustering.predict(times.reshape(-1,1))
    maxima = [np.max(times[np.where(labels == i)]) for i in range(n)]
    sort = np.concatenate((np.sort(maxima)[::-1],np.array([0])))
    result = np.digitize(times,sort)
    return [s[0] for s in scores], result
    
def calculate_linear_scores(scores, n = 5):
    best = min(scores)
    ten_best = best*n
    if not ten_best:
        ten_best = n    
    # apply linear scores:
    m = -n/(ten_best - best)
    b = -1*m*(ten_best)
    
    scores = m*scores+b
    return scores

def calculate_histogram_score(labels, nr_bins = 10, extra_bin_for_min = True):
    labels_copy = labels[labels != -2]
    hist, edges = np.histogram(labels_copy, nr_bins)
    edges_inv = edges[::-1]
    result = np.digitize(labels,edges_inv)
    result[labels == -2] = -1
    if extra_bin_for_min:
        result[labels == min(labels_copy)] = nr_bins+1
    return result

def calculate_agglomerative_score(labels, nr_clusters=10, extra_bin_for_min = True):    
    # Todo: This won't work yet because there is e.g. a max score of 3 for list of length 3 
    
    labels_copy = labels[labels != -2].reshape(-1, 1)
    if len(labels_copy) < nr_clusters:
        nr_clusters = len(labels_copy)
    
    clustering = AgglomerativeClustering(n_clusters = nr_clusters).fit_predict(labels_copy)
    maxima = [np.max(labels_copy[np.where(clustering == i)]) for i in range(nr_clusters)]
    sort = np.concatenate((np.sort(maxima)[::-1],np.array([0])))
    result = np.digitize(labels,sort)
    result[labels == -2] = -1
    if extra_bin_for_min:
        result[labels == min(labels_copy)] = nr_clusters+1
    return result


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
    