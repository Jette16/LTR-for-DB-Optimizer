import torch
import numpy as np

class LTRMetric:
    
    name: str = None
    
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        pass


class AverageBest(LTRMetric):
    
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        assert predicted_y.shape == ground_truth.shape
        predicted_best = ground_truth[torch.arange(predicted_y.shape[0]), torch.argmax(predicted_y, dim = 1)]
        return torch.mean(predicted_best.type(torch.float16))
    
class HitsAtK(LTRMetric):
    def __init__(self, k=10):
        self.k = 10
        
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        assert predicted_y.shape == ground_truth.shape
        
        def func(array):
            split_at = int(len(array)/2)
            return len(np.intersect1d(array[:split_at], array[split_at:]))
        k = self.k if predicted_y.shape[-1] > self.k else predicted_y.shape[-1]
        # get the indices of the best k predicted and true values, row-wise
        top_k_pred = torch.topk(predicted_y, k, dim=-1).indices
        top_k_true = torch.topk(ground_truth, k, dim=-1).indices
        # calculate the number of same elements (row-wise), sum it up and divide it by the number of regarded elements
        temp_all = torch.cat((top_k_pred,top_k_true), 1).detach().numpy()
        return np.sum(np.apply_along_axis(func, 1, temp_all))/torch.numel(top_k_true)
    
class FoundBest(LTRMetric):
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        assert predicted_y.shape == ground_truth.shape
        
        return torch.sum(torch.argmax(predicted_y, dim=1) == torch.argmax(ground_truth, dim=1))
    
class NDCG(LTRMetric):
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        pass
    
class PositionK(LTRMetric):
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        def func(array):
            return np.where(array[:-1] == array[-1])[0]+1
        pred_sort = torch.argsort(predicted_y, descending = True)
        true_sort = torch.argsort(ground_truth, descending = True)[:,0].reshape(-1,1)
        temp_all = torch.cat((pred_sort,true_sort), 1).detach().numpy()
        return np.max(np.apply_along_axis(func, 1, temp_all))
        
class FoundBestK(LTRMetric):
    def __init__(self, k=5):
        self.k = k
    
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        k = self.k if predicted_y.shape[-1] >= self.k else predicted_y.shape[-1] 
        temp_pred = torch.argsort(predicted_y, dim=1, descending=True).detach().numpy()
        temp_true = torch.argsort(ground_truth, dim=1, descending=True).detach().numpy()[:,:k]
        temp_all = np.concatenate((temp_pred,temp_true), 1)
        for row in temp_all:
            for idx, number in enumerate(row[:-k]):
                if number in row[-k:]:
                    return idx+1
    
class TopKFound(LTRMetric):
    def __init__(self, k=5):
        self.k = k
        
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        assert predicted_y.shape == ground_truth.shape
        k = self.k if predicted_y.shape[-1] >= self.k else predicted_y.shape[-1] 
        
        def func(array):
            return len(np.where(array[1:] == array[0])[0])
        
        temp_pred = torch.argmax(predicted_y, dim=1).detach().numpy().reshape(-1,1)
        temp_true = torch.argsort(ground_truth, dim=1, descending=True).detach().numpy()[:,:k]
        temp_all = np.concatenate((temp_pred,temp_true), 1)
        return np.sum(np.apply_along_axis(func, 1, temp_all))
    
    