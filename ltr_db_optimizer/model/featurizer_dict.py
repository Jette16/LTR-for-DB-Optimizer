import os
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation

table_info = TPCHTableInformation()

class FeatureExtractor:
    # We used the following SQL parts:
    # Sort
    # Join: Merge Join, Nested Loop Join, Hash Join
    # Aggregate: Stream Aggregate, Hash Aggregate
    # Scan: Index Scan, Table Scan
    type_vector = ["sort", "stream_aggregate", "hash_aggregate", "merge_join", "nested_loop_join", "hash_join", "index_scan", "table_scan", "null"]
    
    def __init__(self, with_cost = True, small_version=False):
        self.vector_length = len(self.type_vector)
        self.with_cost = with_cost
        self.small_version = small_version
        if with_cost:
            self.vector_length += 1# changed: 2 # for estimated subtree cost and estimated rows
    
    def featurize_query(self, job, rows, folder="./datafarm/output_jobs/"):
        # new: orderby, group_by, nr_joins, estimated_rows, max. relation, min relation
        v = [0,0,0,0,0,0]
        
        with open(folder+job+".pickle", "rb") as f:
            job = pickle.load(f)
        if "WHERE_JOIN" in job:
            v[2] = len(job["WHERE_JOIN"])
        if "GROUP BY" in job:
            v[1] = 1.0
        if "ORDER BY" in job:
            v[0] = 1.0
        v[3] = rows 
        for rel in job["FROM"]:
            row_c = table_info.get_table(rel.split(".")[-1].lower()).row_count
            if row_c > v[4]:
                v[4] = row_c
            if row_c < v[5] or v[5] == 0:
                v[5] = row_c
            
        return v
            
    def featurize_plan(self, plan: dict):
        """
        Featurize plan in form of a dictionary to tree-shaped form for using it in Tree Convolution Layer.
        Recursive approach.
        """
        operation = plan["operator"] 
        none_child = np.zeros(self.vector_length)
        none_child[self.type_vector.index("null")] = 1
        rows = 0
        
        if operation in self.type_vector:
            this_vector = np.zeros(self.vector_length)
            this_vector[self.type_vector.index(operation)] = 1.0
        else:
            # If it is not in type vector, there should also be only one child
            return self.featurize_plan(plan["children"][0])
        if self.with_cost:
            this_vector[-1] = plan["EstimateRows"] 
            rows = plan["EstimateRows"]
            
        # test length of children (2,1,0)
        if len(plan["children"]) == 2:
            # Called for joins
            left_child,_ = self.featurize_plan(plan["children"][0])
            right_child,_ = self.featurize_plan(plan["children"][1])
        elif len(plan["children"]) == 1:
            # Called for sorts and aggregations
            left_child,_ = self.featurize_plan(plan["children"][0])
            right_child = none_child
        else:
            # Called for scans
            return (this_vector, none_child, none_child), rows
        
        return (this_vector, left_child, right_child), rows
    
    def match_cost_plan(self, execution_plan, cost_plan):
        """
        For an execution plan (Python dict) and a cost plan (xml, extracted from SQL Server),
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
                        sub_parts_cost.append(("PhysicalOp", sub_parts[idx+1]))
                    if "EstimateRows" in sub:
                        sub_parts_cost.append(("EstimateRows",float(sub_parts[idx+1])))
                parts_cost.append(sub_parts_cost)

        extended_plan, _ = self.append_features(execution_plan, parts_cost)
        return extended_plan
    
    def append_features(self, execution_plan, label_parts):
        """
        Here, the estimated rows are inserted into the correct child as part of its dictionary.
        Mostly, this function deals with some SQL Server's logic.
        """
        
        if not(execution_plan["operator"] == "top" and execution_plan["children"][0]["operator"] == "sort"):
            curr_part = label_parts[0]
            not_compute_scalar = True
            for c in curr_part:
                if execution_plan["operator"] == "compute_scalar" and c[0] == "PhysicalOp" and not c[1] == "Compute Scalar":
                    not_compute_scalar = False
                    break
                elif c[0] == "PhysicalOp":
                    continue
                execution_plan[c[0]] = c[1]
            if not_compute_scalar:
                label_parts.pop(0)
        for idx, child in enumerate(execution_plan["children"]):
            execution_plan["children"][idx], label_parts = self.append_features(child, label_parts)
        return execution_plan, label_parts

    
def get_features_with_cost_from_folder(plans_folder, cost_folder, return_featurized=True):
    """
    Expects a folder with XML plans (cost_folder) and a folder with execution plans (plans_folder)
    and appends the cost to the respective plan. Additionally, it featurizes the plans.
    """
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
                continue
                
            full_execution_plan = feature_ext.match_cost_plan(execution_plan, cost_plan)

            if return_featurized:
                featurized_plan, rows = feature_ext.featurize_plan(full_execution_plan)
                query_vec = feature_ext.featurize_query_long(job_nr, rows) #delete long
                featurized_trees[file_name] = featurized_plan
                featurized_vecs[file_name] = query_vec
            else:
                featurized_trees[file_name] = full_execution_plan
            
    return featurized_vecs, featurized_trees

def featurize_with_labels(
    plans_folder, 
    cost_folder, 
    label_csv, 
    max_score = 50, 
    score_function = "special",
    extra_for_min = True, 
    special_border = 0.95, 
    normalize = True,
    job_column = "Unnamed: 0.1",
):
    """
    Featurizes the execution plans and calculates the labels for the plans.
    """
    # Featurize the plans
    featurized_vecs, featurized_trees = get_features_with_cost_from_folder(plans_folder, cost_folder)
    
    label_dict = {}
    times = []
        
    df = pd.read_csv(label_csv, index_col = 0)
    # Get the number of the job (current column is "Job0v0_1" with "1" being the ID of the execution plan version)
    df["Job_nr"] = df[job_column].apply(lambda x: x.split("_")[0])
    
    # Iterate over every Job Number
    for job in pd.unique(df["Job_nr"]):
        temp_df = df[df["Job_nr"]==job].copy()
        a = np.array(temp_df["Sum"])
        
        # When dealing with the special score function
        # Calculate the factor for each run time --> result is in [1,+inf)
        if score_function == "special":
            temp = df[df["Job_nr"] == job]
            labels = temp.index
            x = np.array(temp["CPU time"], dtype=np.dtype(float))
            if len(x) == 0:
                continue
            if np.min(x[x>=0]) != 0: # do not regard queries that throw a timeout for calculating the minimum
                x[x>=0] = x[x>=0]/np.min(x[x>=0])
            else:
                # if the minimum is at 0 ms, use x+1 as factors
                x = x + 1 
            times.extend(list(zip(labels,x)))
        else:
            if score_function == "linear":
                a[a<0] = max(a)*10 # Set timeout queries to a value higher 0 
                temp_df["scores"] = calculate_linear_scores(a, n = max_score)
            for idx, row in temp_df.iterrows():
                label_dict[idx] = row["scores"]
    # perform agglomerative clustering to get scores
    if score_function == "special":
        labels, scores = calculate_special_score(times, max_score, special_border)
        for idx, s in enumerate(scores):
            label_dict[labels[idx]] = s
            
    # perform normalization
    if normalize:
        featurized_vecs, featurized_trees, label_dict = normalization(featurized_vecs, featurized_trees, label_dict)
    return featurized_vecs, featurized_trees, label_dict

def normalization(featurized_vecs, featurized_trees, label_dict):
    """
    Function for normalizing the feature vectors of query and plan encoding as well as scores
    """
    
    labels_min = float(min(label_dict.values()))
    labels_max = float(max(label_dict.values()))
    
    # only last col in vector for trees needs to be modified, rest is already one-hot-encoded
    tree_high = 0
    tree_low = 0
    
    # orderby, group_by, nr_joins, estimated_rows, max. relation, min relation
    min_vec = [0]*6
    max_vec = [1, 1, 7, 0, 0, 0]
    
    for key in label_dict.keys():
        tree_high, tree_low = find_tree_high_low(featurized_trees[key], tree_high, tree_low)
        min_vec, max_vec = find_vector_high_low(featurized_vecs[key], min_vec, max_vec)
    
    min_vec = np.array(min_vec)
    max_vec = np.array(max_vec)
    for key in label_dict.keys():
        label_dict[key] = calculate_normalize(label_dict[key], labels_min, labels_max)
        featurized_vecs[key] = list(calculate_normalize(np.array(featurized_vecs[key]), min_vec, max_vec))
        featurized_trees[key] = normalize_tree(featurized_trees[key], tree_low, tree_high)
        
    return featurized_vecs, featurized_trees, label_dict

def calculate_normalize(x, minimum, maximum):
    return (x - minimum) / (maximum - minimum)

def find_vector_high_low(vector, min_vec, max_vec):
    """
    Check if any value in current vector as a higher/lower value than the min or max vec.
    Change it accordingly.
    """
    for i in range(3,len(min_vec)):
        if vector[i] < min_vec[i] or min_vec[i] == 0:
            min_vec[i] = vector[i]
        if vector[i] > max_vec[i]:
            max_vec[i] = vector[i]
    return min_vec, max_vec


def find_tree_high_low(tree, tree_high, tree_low):
    """
    Find the highest and lowest number of estimated rows in this tree and check if that number is
    higher of lower than tree_high/tree_low
    """
    if len(tree) == 3:
        tree_high, tree_low = find_tree_high_low(tree[1], tree_high, tree_low)
        tree_high, tree_low = find_tree_high_low(tree[2], tree_high, tree_low)
    
        if tree[0][-1] > tree_high or tree_high == 0:
            tree_high = tree[0][-1]
        if tree[0][-1] < tree_low or tree_low == 0:
            tree_low = tree[0][-1]
    else:
        if tree[-1] > tree_high or tree_high == 0:
            tree_high = tree[-1]
        if tree[-1] < tree_low or tree_low == 0:
            tree_low = tree[-1]
    
    
    return tree_high, tree_low


def normalize_tree(tree, minimum, maximum):
    """
    Recursive function for normalizing the tree
    """
    if len(tree) == 3:
        curr = tree[0]
        curr[-1] = calculate_normalize(curr[-1], minimum, maximum)
        left = normalize_tree(tree[1], minimum, maximum)
        right = normalize_tree(tree[2], minimum, maximum)
        return (curr, left, right)
    else:
        tree[-1] = calculate_normalize(tree[-1], minimum, maximum)
        return tree


def calculate_special_score(scores, n, border_value):
    """
    Calculate the special scores using Agglomerative clustering. With pruning values above border.
    """
    s = np.array([s[1] for s in scores])
    # Get quantile and set value higher than this value to it
    border = np.quantile(s[s > 1], border_value)
    times = []
    for s in scores:
        if s[1] > border or s[1] < 0:
            times.append(border)    
        else:
            times.append(s[1])
    times = np.array(times)
    temp = times[times > 1]
    
    # Perform agglomerative clustering
    labels = AgglomerativeClustering(n_clusters=n).fit_predict(temp.reshape(-1,1))
    maxima = [np.min(temp[np.where(labels == i)]) for i in range(n)]
    
    # Get the scores of every cluster
    sort = np.concatenate((np.array([border+1]),np.sort(maxima)[::-1],np.array([0])))
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
    scores[scores < 0] = 0
    return scores


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
    