import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models import (split, node, model)
import spyct

def train_model(model, x, y): model.fit(x,y); return model
def model_predict(model, x): return model.predict(x)

def generate_survival_function_data(num_samples, num_days=60, missing_prob=0.1, nan_start_day=50):
    num_features = 5  # Replace with the actual number of features

    X = np.random.rand(num_samples, num_features)

    y = np.zeros((num_samples, num_days))
    for i in range(num_samples):
        survival_prob = np.linspace(1, 0, num_days) * np.random.uniform(0.5, 1.5)

        # Introduce missing values after the specified time point
        nan_start_day = min(nan_start_day, num_days)  # Ensure nan_start_day is within the valid range
        missing_indices = np.arange(nan_start_day, num_days)
        if i%2==0:
            y[i, missing_indices] = np.nan
            y[i, :nan_start_day] = survival_prob[:nan_start_day]
        else:
            y[i, :] = survival_prob

    return X, y


# Example usage:
num_samples = 1000
num_days = 365
num_features = 10

X, y = generate_survival_function_data(num_samples, num_days=num_days)
X, y = torch.Tensor(X), torch.Tensor(y)
print(y)
print("X shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
print(type(X_train), type(y_train))

spyct_ = model.VSpyct(bs=64, max_depth=3)
spyct_ = train_model(spyct_, X_train, y_train)
print(spyct_.num_nodes)
print(spyct_.root_node)
preds = model_predict(spyct_, X_test)
print(preds)
print(preds.shape)

# model_spyct = spyct.Model()
# model_spyct = train_model(model_spyct, X_train.numpy(), y_train.numpy())
# preds_spyct = model_predict(model_spyct, X_test.numpy())
# print(preds_spyct)

i = 1
plt.figure(figsize=(15,5))
plt.plot(preds[i, :, :].mean(axis=0), label='Bayesian SPYCT')
# plt.plot(preds_spyct[i], label = 'SPYCT')

# percentiles_10 = torch.quantile(preds[i, :, :], 0.1, axis=0)
# percentiles_90 = torch.quantile(preds[i, :, :], 0.9, axis=0)

# # Plot the shaded area between the 10th and 90th percentiles
# plt.fill_between(range(preds.shape[-1]), percentiles_10, percentiles_90, color='pink', alpha=0.5)
# plt.legend()
# plt.show()



# import sys
# import os
# sys.path.append('..')
# from src.models import (split, node, model)
# from src.data.dataset import ReducedDataset

# path = os.path.join(os.getcwd(), 'data/raw')
# filename = 'BO_truncated_mso_2018.pcl'
# data = ReducedDataset(path, filename)
# rsf = data.rsf_dataset(to_pcl=False)
# X_train, T_train, E_train, X_test, T_test, E_test = data.rsf_split(to_pcl=False)

# print(type(X_train), type(T_train))

# X_train, T_train, E_train, X_test, T_test, E_test = torch.tensor(np.vstack(X_train.values).astype('float32')), torch.tensor(T_train), E_train, torch.Tensor(np.vstack(X_test.values).astype('float32')), torch.tensor(T_test), E_test
# print(X_train.shape, T_train.shape)

# target_spyct_train = np.zeros((X_train.shape[0],350))
# for key, value in enumerate(T_train.tolist()):
#     for i in range(350):
#         # if(E_train[key]==1):
#         #     target_spyct_train[key][i] = np.nan
#         if(i<value):
#             target_spyct_train[key][i] = 1
#         else:
#             target_spyct_train[key][i] = 0

# target_spyct_test = np.zeros((X_test.shape[0],350))
# for key, value in enumerate(T_test.tolist()):
#     for i in range(350):
#         # if(E_test[key]==1):
#         #     target_spyct_test[key][i] = np.nan
#         if(i<value):
#             target_spyct_test[key][i] = 1
#         else:
#             target_spyct_test[key][i] = 0


# target_spyct_train = torch.Tensor(target_spyct_train)
# target_spyct_test = torch.Tensor(target_spyct_test)

# import pickle
# import spyct
# model_spyct = spyct.Model()
# model_spyct.fit(X_train.numpy(), target_spyct_train.numpy())
# preds_model = model_spyct.predict(X_test.numpy())
# print('Finished with original SPYCT')
# # with open('models/spyct_reduced_v1.pcl', 'wb') as f:
# #     pickle.dump(model_spyct, f)

# bayes_spyct = model.VSpyct(bs=64, max_depth=3)
# print('Training variational SPYCT...')
# bayes_spyct.fit(X_train, target_spyct_train)
# # with open('models/vspyct_reduced_v1.pcl', 'wb') as f:
# #     pickle.dump(bayes_spyct, f)

# print('Predicting...')
# preds_bayes = bayes_spyct.predict(X_test)

# import matplotlib.pyplot as plt

# i = 30
# plt.figure(figsize=(15,5))
# plt.plot(preds_bayes[i, :, :].mean(axis=0), label='Bayes SPYCT')
# plt.plot(preds_model[i], label = 'SPYCT')

# percentiles_10 = torch.quantile(preds_bayes[i, :, :], 0.1, axis=0)
# percentiles_90 = torch.quantile(preds_bayes[i, :, :], 0.9, axis=0)

# # Plot the shaded area between the 10th and 90th percentiles
# plt.fill_between(range(preds_bayes.shape[-1]), percentiles_10, percentiles_90, color='pink', alpha=0.5)
# plt.legend()
# plt.show()