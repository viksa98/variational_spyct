from utils import fix_brier_scores
import sys
sys.path.append('./')
from src.data.dataset import SurvivalDataset
import numpy as np
import statsmodels.api as sm

class BrierScore():
  def __init__(self, data, end_time_point=100):
    from lifelines import CoxPHFitter
    self.data = data
    self.end_time_point = end_time_point
    # Create a Kaplan-Meier estimator
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(data['time'].values, data['status'].values)

    # Calculate the Kaplan-Meier survival curve
    survival_probabilities = kmf.survival_function_
    print(survival_probabilities['KM_estimate'].values.shape)
    # Calculate the inverse probability of censoring weights
    self.censoring_weights = 1 / survival_probabilities

  def brier_score_pysurvival(self, pysurv_model, X_test, T_test, E_test):
    from pysurvival.utils.metrics import brier_score
    example_bs = brier_score(pysurv_model, X_test, T_test, E_test, t_max=None)
    transformed_bs = fix_brier_scores(np.array(example_bs[1]), pysurv_model.times, self.end_time_point)
    return transformed_bs
    

  def brier_score_vspyct():
    pass


if __name__ == '__main__':
  from sksurv.datasets import load_gbsg2
  from sksurv.linear_model import CoxPHSurvivalAnalysis
  from sksurv.metrics import brier_score
  from sksurv.preprocessing import OneHotEncoder
  X, y = load_gbsg2()
  # print(f'X shape: {X.shape}')
  print(f'y shape: {y.shape}')
  # print(y[:2])
  X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
  Xt = OneHotEncoder().fit_transform(X)
  est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)
  survs = est.predict_survival_function(Xt)
  # print(f'Survs shape: {survs.shape}')
  preds = np.array([np.array([fn(x) for x in range(100, 200)]) for fn in survs])
  print(preds.shape)
  times, score = brier_score(y, y, preds, list(range(100, 200)))
  # print(score)
  # import matplotlib.pyplot as plt
  # plt.plot(times, score)
  # plt.show()

  from src.utils import (calculate_bs, fix_predictions, plot_brier)
  from src.data.dataset import SurvivalDataset

  data = SurvivalDataset(fname='data/raw/pbc.rda')
  X_train, T_train, E_train, X_test, T_test, E_test = data.pysurvival_split()

  from pysurvival.models.survival_forest import RandomSurvivalForestModel
  from pysurvival.models.multi_task import LinearMultiTaskModel
  mtlr = LinearMultiTaskModel()
  mtlr.fit(X_train, T_train, E_train, lr=0.0001, l2_reg=1e-2, init_method='zeros')

  predicted_mtlr = mtlr.predict_survival(X_test)
  transformed_predictions_rsf = fix_predictions(predicted_mtlr, mtlr.times, T_test.min(), T_test.max())

  print(transformed_predictions_rsf.shape)

  e_train = [True if x==1 else False for x in E_train]
  e_test = [True if x==1 else False for x in E_test]

  y_test = []

  for x in zip(e_test, T_test): y_test.append((x[0], x[1]))
  y_test = np.array(y_test, dtype=[('status', '?'), ('time', '<f8')])
  
  times, score = brier_score(y_test, y_test, transformed_predictions_rsf, list(range(int(min(y_test.tolist(), key=lambda x: x[1])[1]), int(max(y_test.tolist(), key=lambda x: x[1])[1]))))
  import matplotlib.pyplot as plt
  # times = times-T_test.min()
  print(predicted_mtlr[0])
  print(times)
  print(score)
  print(len(times), len(score))
  print(T_test[0])
  plt.plot(times, score)
  plt.plot(predicted_mtlr[0])
  plt.show()