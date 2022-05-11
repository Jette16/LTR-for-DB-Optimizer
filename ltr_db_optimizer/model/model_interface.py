import math
import torch
import itertools
import numpy as np
import random
import os
import pickle

import torch.nn.functional as F

from ltr_db_optimizer.ext.ptranking.ltr_adhoc.listwise.listnet import ListNet
from ltr_db_optimizer.model.model_structures.comparison_net2 import LTRComparisonNet ##changed
from ltr_db_optimizer.model.metrics import PositionK, HitsAtK,TopKFound, FoundBestK

from ltr_db_optimizer.ext.ptranking.data.data_utils import LABEL_TYPE

def create_model(LossFunction, **kwargs):
    
    class ModelInterface(LossFunction):
        def initialize(self):
            ...
        
        def __init__(self, epochs = 251, sample_size = 25, batch_size = 100, name = "", folder = "", **kwargs):
            super().__init__(sf_para_dict={"sf_id":'pointsf', "opt": None, "lr": None}, **kwargs)
            self.epochs = epochs
            self.sample_size = sample_size
            self.batch_size = batch_size
            self.optimizer = None
            self.device = None
            
            self.name = name
            self.folder = folder
            
            
            # for position k
            self.position_k = PositionK()
            self.found_k = TopKFound(5) # metric defines one of the top k plans was predicted as best plan
            self.best_k_of_k = FoundBestK(5) # metric to find the best position of the top k plans
            
        
        def fit(self, X_train_vecs, X_train_tree, y_train, X_test_vecs, X_test_trees, y_test, use_presort = False, use_scheduler = True, optimizer = "adam"):
            input_dim_1 = 10#9#10
            input_dim_2 = 6 #4 #3# changed: input_dim_2 = 4
            self.net = LTRComparisonNet(input_dim_1, input_dim_2)
            if optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.net.parameters())
            elif optimizer == "adagrad":
                self.optimizer = torch.optim.Adagrad(self.net.parameters())
            elif optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.net.parameters(), lr = 0.1)
            elif optimizer == "rmsprop":
                self.optimizer = torch.optim.RMSprop(self.net.parameters())
                
            
            if use_scheduler:
                scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
            all_losses = []
            avg_scores = []
            test_len = len(list(X_test_trees.keys()))
            
            if not os.path.exists(self.folder+"/"+self.name):
                os.makedirs(self.folder+"/"+self.name)
                
            result_dict = {"epoch":[], "loss": [], "test_best_found": [], "test_top_k_as_best": [],
                           "test_avg_score": [], "test_worst_k": [], "test_avg_k": [] ,
                           "test_worst_k_best_k": [], "test_avg_best_k_of_k": []
                          }
            
            renewed = 0
            best_found = 0
            best_k = 100
            best_avg_k = 100
            best_top_k = 100
            best_avg_top_k = 100
            for epoch in range(self.epochs):
                counter = 0
                losses = 0
                
                job_numbers = list(X_train_vecs.keys())
                curr_job_numbers = random.sample(job_numbers, self.batch_size)
                
                for idx, x in enumerate(curr_job_numbers):
                    curr_keys = list(X_train_vecs[x].keys())
                    length = self.sample_size if len(curr_keys) > self.sample_size else len(curr_keys)
                    random.shuffle(curr_keys)
                    curr_keys = random.sample(curr_keys, length)
                    
                    curr_x = [X_train_tree[x][key] for key in curr_keys]
                    if len(curr_x) == 1:
                        continue
                    curr_x_vec = [X_train_vecs[x][key] for key in curr_keys][0]
                    curr_y = torch.Tensor([[y_train[key] for key in curr_keys]])
                    
                    if torch.count_nonzero(curr_y) == 0:
                        continue
                    
                    counter += 1
                    pred_results = self.net(curr_x_vec, curr_x).t()
                    if use_presort:
                        curr_y, batch_ideal_desc_inds = torch.sort(curr_y, dim=1, descending=True)
                        pred_results = torch.gather(pred_results, dim=1, index=batch_ideal_desc_inds)    

                    loss = self.custom_loss_function(pred_results, curr_y, label_type = LABEL_TYPE.MultiLabel, presort = use_presort)
                    losses += loss
                    
                found = 0
                all_best_k_of_k = 0
                worst_k_best_k = 0
                avg_score_best = 0
                best_k_found = 0
                ignored = 0
                worst = 0
                all_k = 0
                for x_test in X_test_vecs.keys():
                    test_data_vec = list(X_test_vecs[x_test].values())[0]
                    test_data_tree = list(X_test_trees[x_test].values())
                    
                    if not test_data_vec or len(test_data_tree) == 1:
                        ignored += 1
                        continue
                    y_predicted = self.net.predict_all(test_data_vec, test_data_tree)
                    y_predicted_np = y_predicted.detach().numpy()
                    y_true = []
                    for y in X_test_vecs[x_test].keys():
                        y_true.append(y_test[y])
                    if np.argmax(y_predicted_np) == np.argmax(np.array(y_true)):
                        found += 1
                    k = self.position_k.calculate(y_predicted.t(), torch.Tensor([y_true]))
                    best_k_found += self.found_k.calculate(y_predicted.t(), torch.Tensor([y_true]))
                    temp_best_k_of_k = self.best_k_of_k.calculate(y_predicted.t(), torch.Tensor([y_true]))
                    all_best_k_of_k += temp_best_k_of_k
                    all_k += k
                    if k > worst:
                        worst = k
                    # the worst position of the best true top-5 ranked plan
                    if temp_best_k_of_k > worst_k_best_k:
                        worst_k_best_k = temp_best_k_of_k
                    
                    avg_score_best += y_true[np.argmax(y_predicted_np)]
                if counter != 0:    
                    curr_loss = losses/counter
                else:
                    curr_loss = 0
                avg = avg_score_best/(test_len-ignored)
                avg_k = all_k/(test_len-ignored)
                avg_best_k_of_k = all_best_k_of_k/(test_len-ignored)
                renewed += 1
                if avg > best_found:
                    best_found = avg
                    torch.save(self.net.state_dict(), f"{self.folder}/{self.name}/best_avg.pth")
                if best_k > worst:
                    best_k = worst
                    torch.save(self.net.state_dict(), f"{self.folder}/{self.name}/best_k.pth")
                if best_avg_k > avg_k:
                    best_avg_k = avg_k
                    torch.save(self.net.state_dict(), f"{self.folder}/{self.name}/avg_k.pth")
                if avg_best_k_of_k < best_avg_top_k:
                    best_avg_top_k = avg_best_k_of_k
                    torch.save(self.net.state_dict(), f"{self.folder}/{self.name}/avg_best_k_of_k.pth")
                if best_top_k > worst_k_best_k:
                    best_top_k = worst_k_best_k
                    torch.save(self.net.state_dict(), f"{self.folder}/{self.name}/best_top_k.pth")
                    
                
                all_losses.append(curr_loss)
                result_dict["epoch"].append(epoch)
                result_dict["loss"].append(curr_loss)
                result_dict["test_best_found"].append(f"{found}/{test_len-ignored}")
                result_dict["test_top_k_as_best"].append(f"{best_k_found}/{test_len-ignored}")
                result_dict["test_avg_score"].append(avg)
                result_dict["test_worst_k"].append(worst)
                result_dict["test_avg_k"].append(avg_k)
                result_dict["test_worst_k_best_k"].append(worst_k_best_k)
                result_dict["test_avg_best_k_of_k"].append(avg_best_k_of_k)
                
                avg_scores.append(avg)
                
                if epoch % 5 == 0:
                    print(f"Epoch: {epoch} Loss: {curr_loss} Best found Test: {found}/{test_len-ignored} Top k as best: {best_k_found}/{test_len-ignored} Avg. Score Best Test: {avg} Avg k: {avg_k} Worst k: {worst} Avg top k: {avg_best_k_of_k} Worst Best k: {worst_k_best_k}")
                    torch.save(self.net.state_dict(), f"{self.folder}/{self.name}/{epoch}.pth")
                
                #if len(avg_scores) > 100 and all([a <= avg_scores[-101] for a in avg_scores[-100:]]):
                #    print("Early Stopping because no improvement in last 100 epochs")
                #    break
                if use_scheduler:
                    scheduler.step()
            
            with open(self.folder+"/"+self.name+"/info.pickle", "wb") as f:
                pickle.dump(result_dict, f)
            
    return ModelInterface(**kwargs)
          