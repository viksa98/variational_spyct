import torch
from src.models.node import Node, VNode
from src.models.split import learn_split, learn_split_vb
import numpy as np
import pyro

def impurity(values): return torch.sum(torch.var(values, dim=0))

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
                else: node.prototype = torch.mean(target_data[rows], dim=0)

            else: node.prototype = torch.mean(target_data[rows], dim=0)

        self.num_nodes = order

    def predict(self, descriptive_data):
        raw_predictions = [self.root_node.predict(descriptive_data[i]) for i in range(descriptive_data.size(0))]
        return torch.stack(raw_predictions)


class VSpyct:
    def __init__(self, max_depth=np.inf, subspace_size=1, minimum_examples_to_split=2,
                 device='cpu', epochs=150, bs=None, lr=0.001):
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
        self.root_node = VNode(depth=0, root = True)
        splitting_queue = [(self.root_node, rows, total_variance)]
        order = 0
        while splitting_queue:
            node, rows, total_variance = splitting_queue.pop()
            node.order = order
            order += 1
            if total_variance > 0 and node.depth < self.max_depth and rows.size(0) >= self.minimum_examples_to_split:
                split_model, guide = learn_split_vb(
                    rows, descriptive_data[rows], clustering_data[rows],
                    device=self.device, epochs=self.epochs, bs=self.bs, lr=self.lr, subspace_size=self.subspace_size)
                split_model = split_model.linear
                split = split_model(descriptive_data[rows]).squeeze()
                
                rows_right = rows[split > torch.tensor(0., device=self.device)]
                var_right = impurity(clustering_data[rows_right])
                rows_left = rows[split <= torch.tensor(0., device=self.device)]
                var_left = impurity(clustering_data[rows_left])

                if var_left < total_variance or var_right < total_variance:
                    node.split_model = split_model
                    node.guide = guide
                    node.left = VNode(depth=node.depth+1)
                    node.right = VNode(depth=node.depth+1)
                    splitting_queue.append((node.left, rows_left, var_left, ))
                    splitting_queue.append((node.right, rows_right, var_right, ))
                else: node.prototype = torch.mean(target_data[rows], dim=0)

            else: node.prototype = torch.mean(target_data[rows], dim=0)

        self.num_nodes = order

    def predict(self, descriptive_data):
        raw_predictions = [self.root_node.mc_predict(descriptive_data[i]) for i in range(descriptive_data.size(0))]
        return torch.stack(raw_predictions)
