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
import copy

import sklearn.gaussian_process as gp

import model_building.experiment_configuration as ec


class GaussianProcessExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for gaussian processes

    Methods
    -------
    _compute_signature()
        Compute the signature (i.e., an univocal identifier) of this experiment

    _train()
        Performs the actual building of the model

    initialize_regressor()
        Initialize the regressor object for the experiments

    get_default_parameters()
        Get a dictionary with all technique parameters with default values

    repair_hyperparameters()
        Repair and return hyperparameter values which cause the regressor to raise errors
    """
    def __init__(self, campaign_configuration, hyperparameters, regression_inputs, prefix):
        """
        campaign_configuration: dict of str: dict of str: str
            The set of options specified by the user though command line and campaign configuration files

        hyperparameters: dict of str: object
            The set of hyperparameters of this experiment configuration

        regression_inputs: RegressionInputs
            The input of the regression problem to be solved

        prefix: list of str
            The prefix to be added to the signature of this experiment configuration
        """
        super().__init__(campaign_configuration, hyperparameters, regression_inputs, prefix)
        self.technique = ec.Technique.GPR

    def _compute_signature(self, prefix):
        """
        Compute the signature associated with this experiment configuration

        Parameters
        ----------
        prefix: list of str
            The signature of this experiment configuration without considering hyperparameters

        Returns
        -------
            The signature of the experiment
        """
        signature = prefix.copy()
        signature.append("kernel_" + str(self._hyperparameters['kernel']))
        signature.append("alpha_" + str(self._hyperparameters['alpha']))
        signature.append("optimizer_" + str(self._hyperparameters['optimizer']))
        signature.append("n_restarts_optimizer_" + str(self._hyperparameters['n_restarts_optimizer']))
        signature.append("normalize_y_" + str(self._hyperparameters['normalize_y']))
        signature.append("random_state_" + str(self._hyperparameters['random_state']))

        return signature

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        self._logger.debug("Building model for %s", self._signature)
        assert self._regression_inputs
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        self._regressor.fit(xdata, ydata)
        self._logger.debug("Model built")

    def initialize_regressor(self):
        """
        Initialize the regressor object for the experiments
        """
        if not getattr(self, '_hyperparameters', None):
            self._regressor = gp.GaussianProcessRegressor()
        else:
            self._regressor = gp.GaussianProcessRegressor(self._hyperparameters['kernel'],
                                                          alpha=self._hyperparameters['alpha'],
                                                          optimizer=self._hyperparameters['optimizer'],
                                                          n_restarts_optimizer=self._hyperparameters['n_restarts_optimizer'],
                                                          normalize_y=self._hyperparameters['normalize_y'],
                                                          copy_X_train=self._hyperparameters['copy_X_train'],
                                                          # n_targets=self._hyperparameters['n_targets'],
                                                          random_state=self._hyperparameters['random_state'])

    def get_default_parameters(self):
        """
        Get a dictionary with all technique parameters with default values
        """
        return {'kernel': None,
                'alpha': 1e-10, 
                'optimizer': 'fmin_l_bfgs_b', 
                'n_restarts_optimizer': 0, 
                'normalize_y': False, 
                'copy_X_train': True, 
                # 'n_targets': None, 
                'random_state': None}

    def repair_hyperparameters(self, hypers):
        """
        Repair and return hyperparameter values which cause the regressor to raise errors

        Parameters
        ----------
        hypers: dict of str: object
            the hyperparameters to be repaired

        Return
        ------
        dict of str: object
            the repaired hyperparameters
        """
        new_hypers = copy.deepcopy(hypers)
        for key in ['n_restarts_optimizer']:
            new_hypers[key] = int(new_hypers[key])
        return new_hypers
