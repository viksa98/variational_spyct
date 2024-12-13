import torch
from src.models.node import Node, VNode
from src.models.split import learn_split, learn_split_vb
import numpy as np
import pyro
from pyro.infer import Predictive
from sklearn.preprocessing import MinMaxScaler

def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output

# Adjusted function to handle multiple target variables
def multi_target_impurity(tensor, target_dim=1):
    """
    Compute impurity for multiple target variables.
    
    tensor: Input tensor where each row represents a sample and each column represents a different target variable.
    target_dim: The dimension along which the targets are stored (e.g., 1 if columns are different targets).
    """
    # Remove NaN values and check for empty tensor
    if tensor.isnan().all():
        return float('inf')
    
    # Compute variances for each target variable (along target_dim)
    variances = nanvar(tensor, dim=0 if target_dim == 1 else 1)
    
    # Return the sum of variances across all target variables
    total_variance = torch.sum(variances)

    return total_variance

class Spyct:
    def __init__(self, max_depth=np.inf, subspace_size=1, minimum_examples_to_split=2,
                 device='cpu', epochs=100, bs=None, lr=0.1):
        self.minimum_examples_to_split = minimum_examples_to_split
        self.root_node = None
        self.num_nodes = 0
        self.device = device
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.max_depth = max_depth
        self.subspace_size = subspace_size

    def fit(self, descriptive_data, target_data, clustering_data=None, rows=None, enable_mc_dropout=True):
        if clustering_data is None: clustering_data = target_data
        if rows is None: rows = torch.arange(descriptive_data.shape[0])

        self.num_training_instances = descriptive_data.shape[0]
        total_variance = multi_target_impurity(clustering_data)
        self.root_node = Node(depth=0, enable_mc_dropout=enable_mc_dropout)
        splitting_queue = [(self.root_node, rows, total_variance)]
        order = 0
        while splitting_queue:
            node, rows, total_variance = splitting_queue.pop()
            node.order = order
            node.num_instances = rows.shape[0]
            order += 1
            if total_variance > 0 and node.depth < self.max_depth and rows.size(0) >= self.minimum_examples_to_split:
                split_model = learn_split(
                    rows, descriptive_data[rows], clustering_data[rows],
                    device=self.device, epochs=self.epochs, bs=self.bs, lr=self.lr, subspace_size=self.subspace_size)
                split = split_model(descriptive_data[rows]).squeeze()
                rows_right = rows[split > torch.tensor(0., device=self.device)]
                var_right = multi_target_impurity(clustering_data[rows_right])
                rows_left = rows[split <= torch.tensor(0., device=self.device)]
                var_left = multi_target_impurity(clustering_data[rows_left])
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
    
    def feature_importances(self, num_samples=200, k=None):
        non_leaves = []
        def _traverse(node):
            if node is not None:
                if node.left is not None or node.right is not None: non_leaves.append(node)
                _traverse(node.left)
                _traverse(node.right)
        
        _traverse(self.root_node)
        weights = []
        for node in non_leaves:
            weights.append(node.split_model.linear.weight.detach().numpy())
        weights = np.abs(np.array(weights))
        weights = weights.reshape(weights.shape[0], -1)
        weights_normalized = (weights - weights.min()) / (weights.max() - weights.min() + 0.0001)
        importances = torch.zeros((weights_normalized.shape[1]))
        for i, node in enumerate(non_leaves):importances += node.num_instances/self.num_training_instances*(weights_normalized[i, :]/np.linalg.norm(weights_normalized[i, :]))
        if k is not None: return dict(zip(torch.topk(importances, k=k).indices.tolist(), torch.topk(importances, k=k).values.tolist()))
        else: return importances


class VSpyct:
    def __init__(self, max_depth=np.inf, subspace_size=1, minimum_examples_to_split=2,
                 device='cpu', epochs=500, bs=None, lr=0.001):
        self.root_node = None
        self.num_nodes = 0
        self.minimum_examples_to_split = minimum_examples_to_split
        self.device = device
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.max_depth = max_depth
        self.subspace_size = subspace_size
        self.scaler = MinMaxScaler(feature_range=(0, 100))

    def fit(self, descriptive_data, target_data, clustering_data=None, rows=None):
        target_data = torch.Tensor(self.scaler.fit_transform(target_data))
        if clustering_data is None: clustering_data = target_data
        if rows is None: rows = torch.arange(descriptive_data.shape[0])

        self.num_training_instances = descriptive_data.shape[0]
        self.descriptive_data_shape = descriptive_data.shape
        total_variance = multi_target_impurity(clustering_data)
        print(f'Total variance: {total_variance}')
        self.root_node = VNode(descriptive_values=descriptive_data, depth=0, root = True)
        splitting_queue = [(self.root_node, rows, total_variance)]
        order = 0

        while splitting_queue:
            node, rows, total_variance = splitting_queue.pop()
            node.order = order
            node.num_instances = rows.shape[0]
            order += 1
            if total_variance > 0 and node.depth < self.max_depth and rows.size(0) >= self.minimum_examples_to_split:
                split_model, guide, losses = learn_split_vb(
                    rows, descriptive_data[rows], clustering_data[rows],
                    device=self.device, epochs=self.epochs, bs=self.bs, lr=self.lr, subspace_size=self.subspace_size)
                
                predictive = Predictive(model = split_model.linear.to(self.device),
                            guide=guide,
                            num_samples=100,
                            return_sites=("linear.weight", "linear.bias"))
                sdata = predictive(descriptive_data[rows].clone().detach())
                sdata_lin = torch.mean(sdata['linear.weight'], dim=0).reshape(-1, 1).T.T
                sdata_b = torch.mean(sdata['linear.bias'], dim=0).reshape(-1, 1).T
                ssplit = (descriptive_data[rows] @ sdata_lin + sdata_b).sigmoid()
                split = ssplit.reshape(-1)

                # split_model = split_model.linear
                # split = split_model(descriptive_data[rows]).squeeze()
                
                rows_right = rows[split > torch.tensor(0.5, device=self.device)]
                var_right = multi_target_impurity(clustering_data[rows_right])
                rows_left = rows[split <= torch.tensor(0.5, device=self.device)]
                var_left = multi_target_impurity(clustering_data[rows_left])

                print('Rows left: ', rows_left.shape, 'Var left', var_left)
                print('Rows right: ', rows_right.shape, 'Var right', var_right)

                if (var_left < total_variance or var_right < total_variance) and (self.minimum_examples_to_split < rows_right.shape[0] and self.minimum_examples_to_split < rows_left.shape[0]):
                    node.split_model = split_model.linear
                    node.guide = guide
                    node.losses = losses
                    node.left = VNode(descriptive_values=descriptive_data[rows_left], depth=node.depth+1)
                    node.right = VNode(descriptive_values=descriptive_data[rows_right], depth=node.depth+1)
                    splitting_queue.append((node.left, rows_left, var_left, ))
                    splitting_queue.append((node.right, rows_right, var_right, ))
                # potentially to be fixed
                else:
                    if len(target_data.shape)==1: node.prototype = torch.nanmean(target_data[rows], dim=0)
                    else:
                        # node.prototype = torch.nanmean(target_data[rows], dim=0)
                        target_data_np = target_data.numpy()
                        prototype = []
                        for col in range(target_data_np.shape[1]):
                            non_nan_values = target_data_np[rows, col][~torch.isnan(target_data[rows, col])]
                            mean_value = np.nanmean(non_nan_values)
                            prototype.append(mean_value)
                        node.prototype = torch.tensor(np.array(prototype))
            else:
                # node.prototype = torch.nanmean(target_data[rows], dim=0)
                if len(target_data.shape)==1: node.prototype = torch.nanmean(target_data[rows], dim=0)
                else:
                    target_data_np = target_data.numpy()
                    prototype = []
                    for col in range(target_data_np.shape[1]):
                        non_nan_values = target_data_np[rows, col][~torch.isnan(target_data[rows, col])]
                        mean_value = np.nanmean(non_nan_values)
                        prototype.append(mean_value)
                    node.prototype = torch.tensor(np.array(prototype))
        self.num_nodes = order

    def predict(self, descriptive_data):
        raw_predictions = [self.root_node.mc_predict(descriptive_data[i]) for i in range(descriptive_data.size(0))]
        # return torch.stack(raw_predictions)

        # predictions = torch.stack(raw_predictions)
        # predictions = torch.Tensor(self.scaler.inverse_transform(predictions))
        # return predictions
    
        predictions = torch.stack(raw_predictions)

        # Get the original shape (num_examples, target_input_size, monte_carlo_samples)
        original_shape = predictions.shape

        # Reshape predictions to 2D (combine number of examples and target input size)
        # The shape will be (num_examples * target_input_size, monte_carlo_samples)
        predictions_reshaped = predictions.view(-1, original_shape[-1]).cpu().numpy()

        # Apply inverse transform to the 2D reshaped predictions
        predictions_reshaped = self.scaler.inverse_transform(predictions_reshaped)

        # Reshape the transformed predictions back to the original 3D shape
        predictions = torch.Tensor(predictions_reshaped).view(original_shape)
        return predictions
    
    @classmethod
    def create_edge_list(cls, root):
        edges = []
        cls.traverse(root, edges)
        return edges

    @classmethod
    def traverse(cls, node, edges):
        if node is None:
            return

        if node.left is not None:
            edges.append((node, node.left))
            cls.traverse(node.left, edges)

        if node.right is not None:
            edges.append((node, node.right))
            cls.traverse(node.right, edges)

    def get_nodes(self, node, list):
        list.append(node)
        if node.left is not None:
            self.get_nodes(node.left, list)
        if node.right is not None:
            self.get_nodes(node.right, list)

    def get_leaves(self, node, non_leaves):
        if node is not None:
            if node.left is None or node.right is None: non_leaves.append(node)
            self.get_leaves(node.left, non_leaves)
            self.get_leaves(node.right, non_leaves)

    def feature_importances(self, num_samples=100, k=None):
        assert self.num_training_instances!=0, 'The model is not fitted! Unable to calculate feature importances.'
        non_leaves = []
        def _traverse(node):
            if node is not None:
                if node.left is not None or node.right is not None: non_leaves.append(node)
                _traverse(node.left)
                _traverse(node.right)
        
        _traverse(self.root_node)
        weights = []
        for node in non_leaves:
            predictive = Predictive(model=node.split_model.to(self.device),
                                    guide=node.guide,
                                    num_samples=num_samples,
                                    return_sites=(["linear.weight"]))
            data = predictive(node.descriptive_values)['linear.weight']
            weights.append(data.mean(axis=0).numpy())
        weights = np.abs(np.array(weights))
        weights = weights.reshape(weights.shape[0], -1)
        weights_normalized = weights # (weights - weights.min()) / (weights.max() - weights.min() + 0.0001)
        importances = torch.zeros((weights_normalized.shape[1]))
        for i, node in enumerate(non_leaves):
            importances += (node.num_instances/self.num_training_instances)*(weights_normalized[i, :]/(np.linalg.norm(weights_normalized[i, :])+ 0.0001))
        if k is not None: return dict(zip(torch.topk(importances, k=k).indices.tolist(), torch.topk(importances, k=k).values.tolist()))
        else: return importances
        