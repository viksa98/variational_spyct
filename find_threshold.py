import pickle
from sklearn.metrics import mean_absolute_error


import os
import sys
import numpy as np
import pandas as pd

# sys.path.append('..')
from src.models.model import VSpyct

import torch
import pyro
import random

torch.manual_seed(0)
torch.cuda.manual_seed(0)  # Set seed for CUDA
random.seed(0)
np.random.seed(0)


# --- Hyperparams / Setup ---
max_depth = 5
minimum_examples_to_split = 2
epochs = 500
bs = 128
lr = 0.01
subspace_size = 1
device = 'cpu'

# --- Load data ---
with open('cpmp2015_str_files.pcl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

with open('cpmp2015_str_feat_imp.pcl', 'rb') as f:
    feature_importance_scores = pickle.load(f)

feature_names = X_train.columns.tolist()
features_dict = dict(zip(feature_names, feature_importance_scores))

# Sort features by importance ascending so that
# the "least important" feature is first.
sorted_features_least_first = sorted(features_dict, key=features_dict.get)

# --- Define a helper function to train & evaluate ---
def train_and_evaluate_model(X_train, y_train, X_test, y_test, features):
    """
    Subset the training/test data to the given 'features',
    train the VSpyct model, and return the mean_absolute_error on test.
    """
    # Subset columns
    X_train_sub = X_train[features]
    X_test_sub = X_test[features]
    
    # Initialize model
    vspyct = VSpyct(
        bs=bs,
        max_depth=max_depth,
        epochs=epochs,
        lr=lr,
        minimum_examples_to_split=minimum_examples_to_split
    )
    
    # Fit
    vspyct.fit(torch.Tensor(X_train_sub.values), torch.Tensor(y_train.values))
    
    # Predict
    vspyct_preds = vspyct.predict(torch.Tensor(X_test_sub.values))
    vspyct_preds_mean = vspyct_preds.mean(axis=1).numpy()
    
    # Score
    score = mean_absolute_error(y_test.values, vspyct_preds_mean)
    return score

# --- 1) Baseline score with ALL features ---
current_feature_set = list(feature_names)  # start with all features
baseline_score = train_and_evaluate_model(X_train, y_train, X_test, y_test, current_feature_set)
print(f"Baseline MAE: {baseline_score:.4f}")

removed_features = []
scores = [baseline_score]  # keep track of scores at each step

# --- 2) Iteratively remove the least important feature ---
for f in sorted_features_least_first:
    # Create a candidate set that removes 'f'
    candidate_feature_set = [feat for feat in current_feature_set if feat != f]
    
    # Train & evaluate
    candidate_score = train_and_evaluate_model(
        X_train, y_train, 
        X_test, y_test, 
        candidate_feature_set
    )
    
    # Check if removing this feature degrades performance by >10% 
    # i.e. if MAE is 10% higher than the initial baseline
    if candidate_score > baseline_score * 1.10:
        print(
            f"Stopping: Removing '{f}' increases MAE "
            f"by more than 10% of baseline. "
            f"(Candidate MAE: {candidate_score:.4f})"
        )
        break
    else:
        current_feature_set = candidate_feature_set
        removed_features.append(f)
        scores.append(candidate_score)
        print(f"Removed '{f}'; new MAE = {candidate_score:.4f}")

# print("\nSummary:")
# print(f"Features removed: {removed_features}")
# print(f"Remaining features: {current_feature_set}")
# print(f"Number of remaining features: {len(current_feature_set)}")
# print(f"MAE at each step: {scores}")

# --- 3) Save removed features, remaining features, 
# and the ratio of least vs. most important remaining feature ---

# Get importance of the least and most important features in the final set
remaining_importances = [features_dict[feat] for feat in current_feature_set]
lowest_importance = min(remaining_importances)
highest_importance = max(remaining_importances)
importance_ratio = lowest_importance / highest_importance  # float

with open('cpmp_removed_features.pkl', 'wb') as f:
    pickle.dump(removed_features, f)

with open('cpmp_remaining_features.pkl', 'wb') as f:
    pickle.dump(current_feature_set, f)

with open('cpmp_importance_ratio.pkl', 'wb') as f:
    pickle.dump(importance_ratio, f)

print(f"\nSaved pickle files:")
print(f" - removed_features.pkl")
print(f" - remaining_features.pkl")
print(f" - importance_ratio.pkl (value = {importance_ratio:.4f})")
