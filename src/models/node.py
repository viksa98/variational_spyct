import torch
import pyro
import numpy as np
from pyro.infer import Predictive
import random

def load_and_prediction(model, guide, x_test, num_samples = 1, device = 'cpu'):
    pyro.clear_param_store()
    predictive = Predictive(model = model.to(device),
                            guide=guide,
                            num_samples=num_samples,
                            return_sites=("linear.weight", "linear.bias"))
    data = predictive(x_test.clone().detach())#, None,x.shape[0])
    return data

class Node:
    def __init__(self, depth=0, enable_mc_dropout=False):
        self.left = None
        self.right = None
        self.prototype = None
        self.split_model = None
        self.order = None
        self.depth = depth
        self.enable_mc_dropout = enable_mc_dropout

    def predict(self, x):
        if self.is_leaf():
            return self.prototype
        else:
            splits = self.split_model(x, self.enable_mc_dropout)
            print(splits.shape)
            print(f'splits: {splits}')
            if splits.shape[0]>1:
                return torch.stack([self.left.predict(x) if split<=0 else self.right.predict(x) for split in splits])
            else:
                if splits <= 0: return self.left.predict(x)
                else: return self.right.predict(x)

    def is_leaf(self): return self.left is None and self.right is None


class VNode:
    def __init__(self, depth=0, root = False):
        self.left = None
        self.right = None
        self.prototype = None   
        self.split_model = None
        self.guide = None
        self.order = None
        self.depth = depth
        self.is_root = root
        self.param_store = None

    def is_leaf(self): return self.left is None and self.right is None

    def predict(self, x):
        if self.is_leaf():
            return self.prototype
        else:
            data = load_and_prediction(self.split_model, self.guide, x)
            splits = data['linear.weight'] @ x + data['linear.bias']
            splits = splits.reshape(-1)
            if splits.shape[0]>1:
                return torch.stack([self.left.predict(x) if split<=0 else self.right.predict(x) for split in splits])
            else:
                if splits <= 0: return self.left.predict(x)
                else: return self.right.predict(x)

    def mc_predict(self, x, num_samples=30):
        pred_lst = []
        for i in range(num_samples):
            pred_lst.append(self.predict(x))
        # print(len(pred_lst))
        return torch.stack(pred_lst)