"""
Copyright 2021 Bruno Guindani

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

import logging
import os
import pickle
import sys
import pandas as pd
import json

import custom_logger
import sequence_data_processing
import data_preparation.data_loading
import data_preparation.onehot_encoding
from model_building.metrics import Metrics


class Predictor(sequence_data_processing.SequenceDataProcessing):
    """
    Class that uses Pickle objects to make predictions on new datasets
    """
    def __init__(self, regressor_file=None, output_folder="output", debug=False):
        """
        Constructor of the class

        Parameters
        ----------
        regressor_file: str
            Pickle binary file that stores the model to be used for prediction

        output_folder: str
            The directory where all the outputs will be written; it is created by this library and cannot exist before using this module

        debug: bool
            True if debug messages should be printed
        """
        # Set verbosity level and initialize logger
        self.debug = debug
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        self._logger = custom_logger.getLogger(__name__)

        # Initialize flags
        self._output_folder = output_folder
        self._done_file_flag = os.path.join(output_folder, 'done')

        # Read regressor if given
        self._regressor_file = regressor_file
        self._regressor = None

        # Initialize class to compute metrics
        self.metrics = Metrics()


    def load_regressor(self, regressor_file):
        """
        Loads the given regressor as a class member
        """
        with open(regressor_file, "rb") as f:
            self._regressor = pickle.load(f)


    def predict(self, config_file, mape_to_file, regressor_file=None):
        """
        Performs prediction and computes MAPE

        Parameters
        ----------
        config_file: str or dict
            The configuration file describing the experimental campaign to be performed,
            or a dictionary with the same structure

        mape_to_file: bool
            True if computed MAPE should be written to a text file (file name is mape.txt)

        regressor_file: str
            Pickle binary file that stores the model to be used for prediction
        """
        # Check if output path already exist
        if os.path.exists(self._output_folder) and os.path.exists(self._done_file_flag):
            self._logger.error("%s already exists. Terminating the program...", self._output_folder)
            sys.exit(1)
        if not os.path.exists(self._output_folder):
            os.mkdir(self._output_folder)
        
        #Check configuration input type
        if isinstance(config_file,str):
            # Read configuration from the file indicated by the argument
            if not os.path.exists(config_file):
                self._logger.error("%s does not exist", config_file)
                sys.exit(-1)
            # Read config file
            self.load_campaign_configuration(config_file)
        elif isinstance(config_file,dict):
            # Read configuration from the dictionary indicated by the argument
            self._campaign_configuration = config_file
        else:
            print('Unrecognized type for configuration file: '+str(type(config_file)))
            sys.exit(1)

        # Load regressor
        if regressor_file:
            self._regressor_file = regressor_file
        if not self._regressor or regressor_file:
            if 'keras_backend' in self._campaign_configuration['General']:
                backend = self._campaign_configuration['General'].get('keras_backend', 'tensorflow')
                os.environ['KERAS_BACKEND'] = backend
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                import keras
            self.load_regressor(self._regressor_file)
        
        # Read data
        self._logger.info("-->Executing data load")
        data_loader = data_preparation.data_loading.DataLoading(self._campaign_configuration)
        self.data = data_loader.process(None)
        self.data = self.data.data
        self._logger.debug("Current data frame is:\n%s", str(self.data))
        self._logger.info("<--")

        # Start prediction
        self._logger.info("-->Performing prediction")
        yy = self.data[self._campaign_configuration['General']['y']]
        xx = self.data.drop(columns=[self._campaign_configuration['General']['y']])
        yy_pred = self._regressor.predict(xx)

        # Write predictions to file
        yy_both = pd.DataFrame()
        yy_both['real'] = yy
        yy_both['pred'] = yy_pred
        self._logger.debug("Parameters configuration is:")
        self._logger.debug("-->")
        self._logger.debug("Current data frame is:\n%s", str(yy_both))
        self._logger.debug("<--")
        yy_file = os.path.join(self._output_folder, 'prediction.csv')
        with open(yy_file, 'w') as f:
            yy_both.to_csv(f, index=False)
        self._logger.info("Saved to %s", str(yy_file))

        # Compute and output MAPE
        metrics = self.metrics.compute_metrics(yy, yy_pred)
        self._logger.info("---MAPE = %s", str(metrics["MAPE"]))
        if mape_to_file:
          mape_file = os.path.join(self._output_folder, 'metrics.json')
          with open(mape_file, 'w') as f:
            f.write(json.dumps(metrics, indent = 2))
          self._logger.info("Saved MAPE to %s", str(mape_file))

        self._logger.info("<--Performed prediction")

        # Create success flag file
        with open(self._done_file_flag, 'wb') as f:
            pass

    def predict_from_df(self, xx, regressor_file=None):
        """
        Performs prediction on a dataframe

        Parameters
        ----------
        xx: pandas.DataFrame
            The covariate matrix to be used for prediction

        regressor_file: str
            Pickle binary file that stores the model to be used for prediction

        Returns
        -------
        yy_pred
            The predicted values for the dependent variable
        """
        if regressor_file:
            self.load_regressor(regressor_file)

        self._logger.info("-->Performing prediction on dataframe")
        yy_pred = self._regressor.predict(xx)
        self._logger.info("Predicted values are: %s", str(yy_pred))
        self._logger.info("<--Performed prediction")
        return yy_pred
