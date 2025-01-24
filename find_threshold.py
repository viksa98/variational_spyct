import pickle
from sklearn.metrics import mean_absolute_error

import os
import sys
import numpy as np
import pandas as pd
import argparse

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


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate VSpyct model with feature removal.")
    parser.add_argument("num_features_to_remove", type=int, help="Number of least important features to remove.")
    parser.add_argument("data_pickle", type=str, help="Path to the data pickle file.")
    parser.add_argument("feature_importance_pickle", type=str, help="Path to the feature importance pickle file.")
    args = parser.parse_args()

    # Load data
    with open(args.data_pickle, 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    with open(args.feature_importance_pickle, 'rb') as f:
        feature_importance_scores = pickle.load(f)

    feature_names = X_train.columns.tolist()
    features_dict = dict(zip(feature_names, feature_importance_scores))

    # Sort features by importance ascending so that
    # the "least important" feature is first.
    sorted_features_least_first = sorted(features_dict, key=features_dict.get)

    # Validate the number of features to remove
    if args.num_features_to_remove < 0 or args.num_features_to_remove >= len(sorted_features_least_first):
        print(f"Invalid number of features to remove: {args.num_features_to_remove}.")
        print(f"Please enter a value between 0 and {len(sorted_features_least_first) - 1}.")
        sys.exit(1)

    # Select features to keep
    features_to_keep = sorted_features_least_first[args.num_features_to_remove:]
    score = train_and_evaluate_model(X_train, y_train, X_test, y_test, features_to_keep)
    print(f"Removed {args.num_features_to_remove} features; New MAE: {score:.4f}")


if __name__ == "__main__":
    main()
