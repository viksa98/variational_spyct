import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os

from tqdm import trange
import pyro
import pyro.distributions as dist
import pyro.optim
import pyro.infer
import pyro.contrib.autoguide as autoguide
from pyro.contrib.autoguide import AutoDiagonalNormal
import pyro.poutine as poutine
from torch.distributions import constraints
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from pyro.infer.autoguide.guides import AutoDiagonalNormal
from pyro.infer.autoguide.initialization import init_to_mean
from pyro import infer, optim
from pyro.nn.module import to_pyro_module_
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
from pyro.infer import Predictive
import pyro.distributions as dist

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

# def weighted_variance(values, weights, weight_sum):
#     mean = torch.matmul(weights, values) / weight_sum
#     return -torch.sum(mean*mean)

# def weighted_variance(values, weights, weight_sum):
#     values = torch.nan_to_num(values, nan=0.0)
#     weighted_squares = torch.matmul(weights, values) / weight_sum
#     return -torch.sum(weighted_squares*weighted_squares)

def weighted_variance(values, weights, weight_sum):
    values = torch.nan_to_num(values, nan=0.0)
    weights = weights.view(weights.shape[0], 1)
    weighted_squares = weights*values
    return torch.sum((weighted_squares-weighted_squares.mean(axis=0))**2)/ weight_sum

class MC_Dropout_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.2):
        super(MC_Dropout_Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x, enable_mc_dropout=False):
        if not enable_mc_dropout: return self.linear(x)
        else:
            num_samples = 100  # choose the number of samples
            outputs = torch.zeros(num_samples, x.shape[0] if x.dim()>1 else 1, self.output_dim, device=x.device)
            for i in range(num_samples):
                mask = F.dropout(torch.ones_like(x), p=self.dropout_prob, training=enable_mc_dropout)
                outputs[i] = self.linear(x * mask)

            return outputs

class Impurity(PyroModule):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super(Impurity, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = PyroModule[nn.Linear](self.input_dim, self.output_dim)

        self.linear.weight = PyroSample(dist.Normal(0, 1).expand([self.output_dim, self.input_dim]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0, 1).expand([self.output_dim]).to_event(1))

    #fix this with the guide
    def forward(self, descriptive_data, clustering_data):
        # sigma = pyro.sample("sigma", dist.Normal(1., 10.))
        right_selection = self.linear(descriptive_data).reshape(-1).sigmoid()
        left_selection = torch.tensor(1., device=self.device) - right_selection

        right_weight_sum = torch.sum(right_selection)
        left_weight_sum = torch.sum(left_selection)

        var_left = weighted_variance(clustering_data, left_selection, left_weight_sum)
        var_right = weighted_variance(clustering_data, right_selection, right_weight_sum)
        
        impurity = (left_weight_sum * var_left + right_weight_sum * var_right) + torch.norm(self.linear.weight, p=0.5)
        

        #fix this -> impurity cannot be NAN
        try:
            # Define a likelihood term with observed=0
            with pyro.plate("data", descriptive_data.shape[0]):
                # registriraj ja varijansata kako parametar
                obs = pyro.sample("obs", dist.Normal(impurity, 1.0), obs=torch.tensor(0.))
        except: print('NAN!')
        
        
        return impurity

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def is_converged(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
        
def learn_split(rows, descriptive_data, clustering_data, device, epochs, bs, lr, subspace_size):
    selected_attributes = np.random.choice(a=[False, True],
                                           size=descriptive_data.size(1),
                                           p=[1-subspace_size, subspace_size])
    descriptive_subset = descriptive_data[:, selected_attributes]
    # model = torch.nn.Linear(in_features=descriptive_subset.size(1), out_features=1).to(device)
    model = MC_Dropout_Linear(input_dim=descriptive_subset.size(1), output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if bs is None: bs = rows.size(0)
    num_batches = math.ceil(rows.size(0) / bs)

    for _ in range(epochs):
        for b in range(num_batches):
            descr = descriptive_subset[b*bs:(b+1)*bs]
            clustr = clustering_data[b*bs:(b+1)*bs]
            right_selection = model(descr).reshape(-1).sigmoid()
            left_selection = torch.tensor(1., device=device) - right_selection
            right_weight_sum = torch.sum(right_selection)
            left_weight_sum = torch.sum(left_selection)
            var_left = weighted_variance(clustr, left_selection, left_weight_sum)
            var_right = weighted_variance(clustr, right_selection, right_weight_sum)
            impurity = (left_weight_sum * var_left + right_weight_sum * var_right) + torch.norm(model.linear.weight, p=0.5)
            impurity.backward()
            optimizer.step()
            model.zero_grad()

    return model.to(device).eval()

def learn_split_vb(rows, descriptive_data, clustering_data, device, epochs, bs, lr, subspace_size, patience=4, enable_validation=False):
    pyro.enable_validation(enable_validation)
    selected_attributes = np.random.choice(a=[False, True],
                                           size=descriptive_data.size(1),
                                           p=[1-subspace_size, subspace_size])
    descriptive_subset = descriptive_data[:, selected_attributes]

    pyro.clear_param_store()
    model = Impurity(input_dim=descriptive_subset.size(1), output_dim=1).to(device)
    print(model)
    guide = AutoDiagonalNormal(model)
    adam_params = {"lr": lr }#, "betas": (0.90, 0.999)}
    optimizer = pyro.optim.Adam(adam_params)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.1)

    # Setup SVI
    svi = infer.SVI(model,
                    guide,
                    optimizer,
                    loss=infer.TraceMeanField_ELBO(num_particles=10))
    losses = []
    _steps = 0
    if bs is None: bs = rows.size(0)
    num_batches = math.ceil(rows.size(0) / bs)
    for epoch in trange(epochs, desc="Epochs"):
        train_loss = 0.0
        for b in range(num_batches):
            descr = descriptive_subset[b*bs:(b+1)*bs]
            clustr = clustering_data[b*bs:(b+1)*bs]
            train_loss += svi.step(descr, clustr)
            if _steps<1:
                param_store = pyro.get_param_store()['AutoDiagonalNormal.loc']
            _steps+=1
        if early_stopping.is_converged(train_loss):
            print(f"Early stopping at epoch {epoch}.")
            break
        losses.append(train_loss)
        print("[iteration %04d] loss: %.4f" % (epoch+1, train_loss))
    print(model)
    return (model, guide, param_store)