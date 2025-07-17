Bayes_spyct is a research prototype that extends Oblique Predictive Clustering Trees (SPYCTs) with full Bayesian treatment and Monte-Carlo uncertainty estimation.

It provides:

* `Spyct`  – a fast, deterministic variant trained with stochastic gradient descent.
* `VSpyct` – a Bayesian version that learns a posterior distribution over split parameters using variational inference.  At prediction time the model performs Monte-Carlo sampling which allows you to inspect both mean predictions and their epistemic uncertainty.

Although the original motivation was **survival analysis**, the models can be applied to any tabular regression / multi-output task.

## Installation

```bash
# clone the repository
$ git clone https://github.com/your_username/bayes_spyct.git
$ cd bayes_spyct

# create a virtual-env and install dependencies
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# install the library in editable mode
$ pip install -e .
```

## Quick start

The snippet below shows the minimal steps required to fit a Bayesian tree and obtain both predictions and their standard deviation:

```python
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Import the Bayesian tree
from src.models.model import VSpyct

# 1️⃣  Prepare some toy data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# 2️⃣  Instantiate and fit the model
model = VSpyct(bs = 256,          # batch size
                lr = 0.001,       # learning rate
                max_depth=5,      # optional – limit tree depth
               epochs=300)        # number of variational-inference steps
model.fit(X_train, y_train)

# 3️⃣  Predict with uncertainty (100 MC samples by default)
predictions = model.predict(X_test)  # shape: (n_samples, 1, mc_samples)

# Aggregate across Monte-Carlo dimension
mean_pred = predictions.mean(dim=-1)      # point estimate
std_pred  = predictions.std(dim=-1)       # epistemic uncertainty

print(mean_pred[:5])
print(std_pred[:5])
```

For a deterministic tree (faster but without uncertainty) simply replace `VSpyct` with `Spyct` and drop the aggregation over Monte-Carlo samples.