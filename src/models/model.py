import torch
from src.models.node import Node, VNode
from src.models.split import learn_split, learn_split_vb
import numpy as np
import pyro
from pyro.infer import Predictive

def nanvar(y, dim=0):
    mean = torch.nanmean(y, dim)
    squared_diff = (y - mean.unsqueeze(dim)) ** 2
    valid_count = torch.sum(~torch.isnan(y), dim)
    variance = torch.nansum(squared_diff, dim) / (valid_count - 1) 
    return variance

def impurity(values): return torch.sum(nanvar(values, dim=0))

class Spyct:
    def __init__(self, max_depth=np.inf, subspace_size=1, minimum_examples_to_split=2,
                 device='cpu', epochs=200, bs=None, lr=0.01):
        self.minimum_examples_to_split = minimum_examples_to_split
        self.root_node = None
        self.num_nodes = 0
        self.device = device
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.max_depth = max_depth
        self.subspace_size = subspace_size

    def fit(self, descriptive_data, target_data, clustering_data=None, rows=None):
        if clustering_data is None: clustering_data = target_data
        if rows is None: rows = torch.arange(descriptive_data.shape[0])

        total_variance = impurity(clustering_data)
        self.root_node = Node(depth=0, enable_mc_dropout=True)
        splitting_queue = [(self.root_node, rows, total_variance)]
        order = 0
        while splitting_queue:
            node, rows, total_variance = splitting_queue.pop()
            node.order = order
            order += 1
            if total_variance > 0 and node.depth < self.max_depth and rows.size(0) >= self.minimum_examples_to_split:
                split_model = learn_split(
                    rows, descriptive_data[rows], clustering_data[rows],
                    device=self.device, epochs=self.epochs, bs=self.bs, lr=self.lr, subspace_size=self.subspace_size)
                split = split_model(descriptive_data[rows]).squeeze()
                rows_right = rows[split > torch.tensor(0., device=self.device)]
                var_right = impurity(clustering_data[rows_right])
                rows_left = rows[split <= torch.tensor(0., device=self.device)]
                var_left = impurity(clustering_data[rows_left])
                if var_left < total_variance or var_right < total_variance:
                    node.split_model = split_model
                    node.left = Node(depth=node.depth+1)
                    node.right = Node(depth=node.depth+1)
                    splitting_queue.append((node.left, rows_left, var_left, ))
                    splitting_queue.append((node.right, rows_right, var_right, ))
                else: node.prototype = torch.nanmean(target_data[rows], dim=0)

            else: node.prototype = torch.nanmean(target_data[rows], dim=0)

        self.num_nodes = order

    def predict(self, descriptive_data):
        raw_predictions = [self.root_node.predict(descriptive_data[i]) for i in range(descriptive_data.size(0))]
        return torch.stack(raw_predictions)


class VSpyct:
    def __init__(self, max_depth=np.inf, subspace_size=1, minimum_examples_to_split=2,
                 device='cpu', epochs=500, bs=None, lr=0.001):
        self.minimum_examples_to_split = minimum_examples_to_split
        self.root_node = None
        self.num_nodes = 0
        self.device = device
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.max_depth = max_depth
        self.subspace_size = subspace_size

    def fit(self, descriptive_data, target_data, clustering_data=None, rows=None):
        if clustering_data is None: clustering_data = target_data
        if rows is None: rows = torch.arange(descriptive_data.shape[0])

        self.num_training_instances = descriptive_data.shape[0]
        self.descriptive_data_shape = descriptive_data.shape
        total_variance = impurity(clustering_data)
        print(f'Total variance: {total_variance}')
        self.root_node = VNode(depth=0, root = True)
        splitting_queue = [(self.root_node, rows, total_variance)]
        order = 0

        while splitting_queue:
            node, rows, total_variance = splitting_queue.pop()
            node.order = order
            node.num_instances = rows.shape[0]
            order += 1
            if total_variance > 0 and node.depth < self.max_depth and rows.size(0) >= self.minimum_examples_to_split:
                if order<=1:
                    split_model, guide, param_store = learn_split_vb(
                        rows, descriptive_data[rows], clustering_data[rows],
                        device=self.device, epochs=self.epochs, bs=self.bs, lr=self.lr, subspace_size=self.subspace_size)
                    self.root_node.param_store = param_store
                else:
                    split_model, guide, _ = learn_split_vb(
                        rows, descriptive_data[rows], clustering_data[rows],
                        device=self.device, epochs=self.epochs, bs=self.bs, lr=self.lr, subspace_size=self.subspace_size)
                predictive = Predictive(model = split_model.linear.to(self.device),
                            guide=guide,
                            num_samples=50,
                            return_sites=("linear.weight", "linear.bias"))
                sdata = predictive(descriptive_data[rows].clone().detach())
                sdata_lin = torch.mean(sdata['linear.weight'], dim=0).reshape(-1, 1).T.T
                sdata_b = torch.mean(sdata['linear.bias'], dim=0).reshape(-1, 1).T
                ssplit = descriptive_data[rows] @ sdata_lin + sdata_b
                split = ssplit.reshape(-1)

                # split_model = split_model.linear
                # split = split_model(descriptive_data[rows]).squeeze()
                
                rows_right = rows[split > torch.tensor(0., device=self.device)]
                var_right = impurity(clustering_data[rows_right])
                rows_left = rows[split <= torch.tensor(0., device=self.device)]
                var_left = impurity(clustering_data[rows_left])

                print('Var left', var_left)
                print('Var right', var_right)

                if var_left < total_variance or var_right < total_variance:
                    node.split_model = split_model.linear
                    node.guide = guide
                    node.left = VNode(depth=node.depth+1)
                    node.right = VNode(depth=node.depth+1)
                    splitting_queue.append((node.left, rows_left, var_left, ))
                    splitting_queue.append((node.right, rows_right, var_right, ))
                # potentially to be fixed
                else: node.prototype = torch.nanmean(target_data[rows], dim=0)
            # potentially to be fixed
            else: node.prototype = torch.nanmean(target_data[rows], dim=0)
        self.num_nodes = order

    def predict(self, descriptive_data):
        raw_predictions = [self.root_node.mc_predict(descriptive_data[i]) for i in range(descriptive_data.size(0))]
        return torch.stack(raw_predictions)

    def feature_importances(self, k=None):
        assert self.num_training_instances!=0, 'The model is not fitted! Unable to calculate feature importances.'
        non_leaves = []
        def traverse(node):
            if node is not None:
                if node.left is not None or node.right is not None: non_leaves.append(node)
                traverse(node.left)
                traverse(node.right)
        
        traverse(self.root_node)
        weights = []
        for node in non_leaves:
            predictive = Predictive(model=node.split_model.to(self.device),
                                    guide=node.guide,
                                    num_samples=200,
                                    return_sites=(["linear.weight"]))
            data = predictive(torch.ones(self.descriptive_data_shape[1]))['linear.weight']
            weights.append(data.mean(axis=0).numpy())
        weights = np.array(weights)
        weights = weights.reshape(weights.shape[0], -1)
        weights_normalized = (weights - weights.min()) / (weights.max() - weights.min() + 0.0001)
        importances = torch.zeros((weights_normalized.shape[1]))
        for i, node in enumerate(non_leaves): importances += node.num_instances/self.num_training_instances*(weights_normalized[i, :]/np.linalg.norm(weights_normalized[i, :]))
        if k is not None: return dict(zip(torch.topk(importances, k=k).indices.tolist(), torch.topk(importances, k=k).values.tolist()))
        else: return importances
        