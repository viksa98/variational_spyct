import sys
sys.path.append('.')
from src.utils import fix_predictions
from src.data.dataset import SurvivalDataset
import numpy as np
import torch

class IPCWBrier:
  def __init__(self, times, event_observed):
    from lifelines import KaplanMeierFitter
    self.times = times
    self.event_observed = event_observed
    self.num_events = sum(event_observed)
    self.kmf = KaplanMeierFitter()
    self.kmf.fit(self.times, self.event_observed)
    cumulative_censoring_probs = 1 - self.kmf.survival_function_.values
    self.ipcw_coeffs = (1 / cumulative_censoring_probs)[1:]

  def evaluate(self, times_arr, predicted_probs):
    if isinstance(times_arr, torch.Tensor): times_arr = times_arr.numpy()
    if isinstance(predicted_probs, torch.Tensor): predicted_probs = predicted_probs.numpy()
    brier_score = torch.nanmean(torch.tensor(self.ipcw_coeffs) * torch.tensor(np.power(times_arr - predicted_probs, 2)), axis=0) / self.num_events
    return brier_score


if __name__ == '__main__':
  data = SurvivalDataset(fname='pbc.rda', path='data/raw/')
  _, _, y_train, y_test = data.get_tensors()
  X_train, T_train, E_train, X_test, T_test, E_test = data.pysurvival_split()
  from pysurvival.models.multi_task import LinearMultiTaskModel
  mtlr = LinearMultiTaskModel()
  mtlr.fit(X_train, T_train, E_train, lr=0.0001, l2_reg=1e-2, init_method='zeros')
  predicted_mtlr = mtlr.predict_survival(X_test)
  transformed_predictions_mtlr = fix_predictions(predicted_mtlr, mtlr.times, T_train.max())
  ipcw_brier = IPCWBrier(T_test, E_test)
  bs_mtlr = ipcw_brier.evaluate(y_test, transformed_predictions_mtlr)
  import matplotlib.pyplot as plt
  plt.plot(bs_mtlr)
  plt.show()