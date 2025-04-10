"""
Copyright 2019 Marco Lattuada
Copyright 2025 Federica Filippini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = np.finfo(np.float64).eps
    if len(y_true.shape) == 1:
        y_true = y_true.values.reshape(y_true.shape[0],1)
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.average(np.average(mape, axis=0), axis = 0)


class Metrics:
    def __init__(self):
        self._metrics_dict = {
            "MAPE": {
              "func": mean_absolute_percentage_error,
              "attributes": {},
              "comp": (lambda x,y : x < y)  # lower is better
            }, 
            "RMSE": {
              "func": mean_squared_error,
              "attributes": {"squared": False},
              "comp": (lambda x,y : x < y)  # lower is better
            }, 
            "R^2": {
              "func": r2_score,
              "attributes": {},
              "comp": (lambda x,y : x > y)  # greater is better
            }, 
            "MAE": {
              "func": mean_absolute_error,
              "attributes": {},
              "comp": (lambda x,y : x < y)  # lower is better
            }, 
            "MSE": {
              "func": mean_squared_error,
              "attributes": {"squared": True},
              "comp": (lambda x,y : x < y)  # lower is better
            }
        }
    
    def compute_metric(self, metric, real_y, predicted_y):
        if metric in self._metrics_dict:
            return self._metrics_dict[metric]["func"](real_y, predicted_y, **self._metrics_dict[metric]["attributes"])
        else:
            return None
    
    def compute_metrics(self, real_y, predicted_y):
        metrics = {}
        for metric in self._metrics_dict:
            metrics[metric] = self.compute_metric(metric, real_y, predicted_y)
        return metrics
    
    def greater_is_better(self, metric):
        if metric in self._metrics_dict:
            return self._metrics_dict[metric]["comp"](3,2)
        else:
            return None
    
    def get_metric_operator(self, metric):
        if metric in self._metrics_dict:
            return self._metrics_dict[metric]["func"]
        else:
            return None
    
    def get_comparison_operator(self, metric):
        if metric in self._metrics_dict:
            return self._metrics_dict[metric]["comp"]
        else:
            return None
    
    def supported_metrics(self):
        return list(self._metrics_dict.keys())
