"""
Copyright 2019 Marco Lattuada
Copyright 2021 Bruno Guindani
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

import collections
import logging
import multiprocessing
import os
import sys
from typing import Dict
from typing import List

import tqdm

import custom_logger
import model_building.experiment_configuration as ec
from model_building.metrics import Metrics


def evaluate_wrapper(experiment_configuration):
    if experiment_configuration.trained:
        experiment_configuration.evaluate()
    return experiment_configuration


def recursivedict():
    return collections.defaultdict(recursivedict)


class Results:
    """
    Class collecting all the results of a campaign

    Attributes
    ----------
    _campaign_configuration: dict of dict:
        The set of options specified by the user though command line and 
        campaign configuration files


    _exp_confs : List[ec.ExperimentConfiguration]
        The list of all the experiments

    raw_results : Dict[str, Dict[str, Dict[str, float]]]
        All the raw results; first key is the signature of the experiment; 
        second key is the metric (e.g., MAPE), third key is the dataset 
        (i.e., training, hp_selection, validation)

    Methods
    -------
    collect_data()
        Collect the data of all the considered experiment configurations

    get_bests()
        Compute the best overall method
    """
    def __init__(
            self, 
            campaign_configuration, 
            exp_confs: List[ec.ExperimentConfiguration]
        ):
        """
        Parameters
        ----------
        campaign_configuration: dict of dict:
            The set of options specified by the user though command line and 
            campaign configuration files

        exp_confs: List[ec.ExperimentConfiguration]
            The list of the run experiment configurations
        """
        self._campaign_configuration = campaign_configuration
        self.techniques = campaign_configuration['General']['techniques']
        self.metric = campaign_configuration['General']['metric']
        self._metrics_calculator = Metrics()
        self._comparison_operator = self._metrics_calculator.get_comparison_operator(self.metric)
        self._exp_confs = exp_confs
        self.raw_results: Dict[str, Dict[str, Dict]] = {}
        self._logger = custom_logger.getLogger(__name__)

        # Logger writes to stdout and file
        self.file_handler = logging.FileHandler(os.path.join(self._campaign_configuration['General']['output'], 'results.txt'), 'a+')
        self._logger.addHandler(self.file_handler)

    def dismiss_handler(self):
        self._logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def collect_data(self):
        """
        Collect the data of all the performed experiments
        """
        exp_conf: ec.ExperimentConfiguration
        
        processes_number = self._campaign_configuration['General']['j']
        if processes_number == 1:
            self._logger.info("-->Evaluate experiments (sequentially) -- chosen metric: %s", self.metric)
            for exp_conf in tqdm.tqdm(self._exp_confs, dynamic_ncols=True):
                if not exp_conf.trained:
                    continue
                exp_conf.evaluate()
            self._logger.info("<--")
        else:
            self._logger.info("-->Evaluate experiments (in parallel) -- chosen metric: %s", self.metric)
            with multiprocessing.Pool(processes_number) as pool:
                self._exp_confs = list(tqdm.tqdm(pool.imap(evaluate_wrapper, self._exp_confs), total=len(self._exp_confs)))
                self._logger.info("<--")

        self.raw_results = {}
        for exp_conf in self._exp_confs:
            self.raw_results[tuple(exp_conf.get_signature())] = exp_conf.metrics

    def get_bests(self):
        """
        Identify for each considered technique, the configuration with the best validation error according to the required metric, also recover all other metrics, and print the results

        Returns
        -------
        an ExperimentConfiguration instance
            the best-performing experiment configuration in terms of the required metric
        a Technique(Enum) instance
            Enum object indicating the technique of the best-performing experiment configuration (see experiment_configuration.py)
        """
        set_names = ["training", "hp_selection", "validation"]
        run_tec_conf_set = recursivedict()
        validation = self._campaign_configuration['General']['validation']
        hp_selection = self._campaign_configuration['General']['hp_selection']
        if (validation, hp_selection) in {("All", "All"), ("Interpolation", "All"), ("Extrapolation", "All"), ("All", "HoldOut"), ("HoldOut", "All"), ("HoldOut", "HoldOut"), ("Interpolation", "HoldOut"), ("Extrapolation", "HoldOut")}:
            # For each run, for each technique the best configuration
            run_tec_best_conf = recursivedict()

            # Hyperparameter search
            for conf in self._exp_confs:
                if not conf.trained:
                    continue
                run = int(conf.get_signature()[0].replace("run_", ""))
                technique = conf.technique
                run_tec_conf_set[run][technique][str(conf.get_signature()[4:])][self.metric] = conf.metrics[self.metric]
                # First experiment for this technique or better than the current best
                if technique not in run_tec_best_conf[run] or self._comparison_operator(run_tec_conf_set[run][technique][str(conf.get_signature()[4:])][self.metric]["hp_selection"], run_tec_best_conf[run][technique].metrics[self.metric]["hp_selection"]):
                    run_tec_best_conf[run][technique] = conf

            # Print results for each run
            for run in range(0, self._campaign_configuration['General']['run_num']):
                unused_techniques = self.techniques
                self._logger.info("Printing results for run %s", run)
                self._logger.info("-->Metrics for all techniques:")
                overall_run_best = None
                # Print data of single techniques
                padding = max([len(str(t)) for t in run_tec_best_conf[run]])
                for technique in run_tec_best_conf[run]:
                    temp = run_tec_best_conf[run][technique]
                    if ec.enum_to_configuration_label[technique] in unused_techniques:
                        unused_techniques.remove(ec.enum_to_configuration_label[technique])
                    printed_name = str(technique).ljust(padding)
                    if bool(self._campaign_configuration['General']['details']):
                        for metric, metric_val in temp.metrics.items():
                            self._logger.info("%s [%s]: (Training %f - HP Selection %f) - Validation %f", printed_name, metric, metric_val["training"], metric_val["hp_selection"], metric_val["validation"])
                    else:
                        self._logger.info("%s [%s]: (Training %f - HP Selection %f) - Validation %f", printed_name, self.metric, temp.metrics[self.metric]["training"], temp.metrics[self.metric]["hp_selection"], temp.metrics[self.metric]["validation"])

                    # Compute which is the best technique
                    if not overall_run_best or self._comparison_operator(temp.metrics[self.metric]["hp_selection"], overall_run_best.metrics[self.metric]["hp_selection"]):
                        overall_run_best = temp
                if not overall_run_best:
                    self._logger.error("No valid model was found")
                    exit(1)
                if unused_techniques:
                    self._logger.info("The following techniques had no successful runs: %s", str(unused_techniques))
                self._logger.info("<--Overall best result (according to %s) is %s, with configuration %s", self.metric, overall_run_best.technique, overall_run_best.get_signature()[4:])
                self._logger.info("Metrics for best result:")
                self._logger.info("-->")
                for metric in temp.metrics:
                    metric_val = overall_run_best.metrics[metric]
                    self._logger.info("%s: (Training %f - HP Selection %f) - Validation %f", metric, metric_val["training"], metric_val["hp_selection"], metric_val["validation"])
                self._logger.info("<--")

        elif (validation, hp_selection) in {("KFold", "All"), ("KFold", "HoldOut")}:
            folds = float(self._campaign_configuration['General']['folds'])
            # For each run, for each fold, for each technique, the best configuration
            run_fold_tec_best_conf = recursivedict()

            # Hyperparameter search inside each fold
            for conf in self._exp_confs:
                if not conf.trained:
                    continue
                run = int(conf.get_signature()[0].replace("run_", ""))
                fold = int(conf.get_signature()[1].replace("f", ""))
                technique = conf.technique
                if self.metric not in run_tec_conf_set[run][technique][str(conf.get_signature_string()[4:])]:
                    for metric in conf.metrics:
                        for set_name in set_names:
                            run_tec_conf_set[run][technique][str(conf.get_signature_string()[4:])][metric][set_name] = 0
                for metric, metric_vals in conf.metrics.items():
                    for set_name in set_names:
                        run_tec_conf_set[run][technique][str(conf.get_signature_string()[4:])][metric][set_name] += conf.metrics[self.metric][set_name] / folds
                # First experiment for this fold+technique or better than the current best
                if technique not in run_fold_tec_best_conf[run][fold] or self._comparison_operator(conf.metrics[self.metric]["hp_selection"], run_fold_tec_best_conf[run][fold][technique].metrics[self.metric]["hp_selection"]):
                    run_fold_tec_best_conf[run][fold][technique] = conf

            # Aggregate different folds (only the value of the metrics)
            run_tec_set = recursivedict()
            for run in run_fold_tec_best_conf:
                for fold in run_fold_tec_best_conf[run]:
                    for tec in run_fold_tec_best_conf[run][fold]:
                        if self.metric not in run_tec_set[run][technique]:
                            for metric in run_fold_tec_best_conf[run][fold][tec].metrics:
                                for set_name in set_names:
                                    run_tec_set[run][tec][metric][set_name] = 0
                        for metric, metric_vals in run_fold_tec_best_conf[run][fold][tec].metrics.items():
                            for set_name in set_names:
                                run_tec_set[run][tec][metric][set_name] = metric_vals[set_name]

            # Print results for each run
            for run in range(0, self._campaign_configuration['General']['run_num']):
                unused_techniques = self.techniques
                self._logger.info("Printing results for run %s", run)
                self._logger.info("-->Metrics for all techniques:")
                overall_run_best = ()  # (technique, metrics)

                # Print data of single techniques
                padding = max([len(str(t)) for t in run_tec_set[run]])
                for technique in run_tec_set[run]:
                    if ec.enum_to_configuration_label[technique] in unused_techniques:
                        unused_techniques.remove(ec.enum_to_configuration_label[technique])
                    printed_name = str(technique).ljust(padding)
                    if bool(self._campaign_configuration['General']['details']):
                        for metric, metric_val in run_tec_set[run][technique].items():
                            self._logger.info("%s [%s]: (Training %f - HP Selection %f) - Validation %f", printed_name, metric, metric_val["training"], metric_val["hp_selection"], metric_val["validation"])
                    else:
                        metric_val = run_tec_set[run][technique][self.metric]
                        self._logger.info("%s [%s]: (Training %f - HP Selection %f) - Validation %f", printed_name, self.metric, metric_val["training"], metric_val["hp_selection"], metric_val["validation"])

                    # Compute which is the best technique
                    if not overall_run_best or self._comparison_operator(run_tec_set[run][technique][self.metric]["hp_selection"], overall_run_best[1][self.metric]["hp_selection"]):
                        overall_run_best = (technique, run_tec_set[run][technique])

                if not overall_run_best:
                    self._logger.error("No valid model was found")
                    exit(1)
                if unused_techniques:
                    self._logger.info("The following techniques had no successful runs: %s", str(unused_techniques))
                self._logger.info("<--Overall best result (according to %s) is %s", self.metric, overall_run_best[0])
                self._logger.info("Metrics for best result:")
                self._logger.info("-->")
                for metric, metric_val in overall_run_best[1].items():
                    self._logger.info("%s: (Training %f - HP Selection %f) - Validation %f", metric, metric_val["training"], metric_val["hp_selection"], metric_val["validation"])
                self._logger.info("<--")

            # Overall best will contain as first argument the technique with the best (across runs) average (across folds) metric on validation; now we consider on all the runs and on all the folds the configuraiton of this technique with best validation metric

        elif (validation, hp_selection) in {("All", "KFold"), ("HoldOut", "KFold"), ("Interpolation", "KFold"), ("Extrapolation", "KFold")}:
            folds = float(self._campaign_configuration['General']['folds'])
            # For each run, for each technique, for each configuration, the aggregated metrics
            run_tec_conf_set = recursivedict()

            # Hyperparameter search aggregating over folders
            for conf in self._exp_confs:
                if not conf.trained:
                    continue
                run = int(conf.get_signature()[0].replace("run_", ""))
                fold = int(conf.get_signature()[2].replace("f", ""))
                technique = conf.technique
                configuration = str(conf.get_signature()[4:])
                if self.metric not in run_tec_conf_set[run][technique][configuration]:
                    for metric in conf.metrics:
                        for set_name in set_names:
                            run_tec_conf_set[run][technique][configuration][metric][set_name] = 0
                for metric, metric_vals in conf.metrics.items():
                    for set_name in set_names:
                        run_tec_conf_set[run][technique][configuration][metric][set_name] += metric_vals[set_name] / folds

            # Select the best configuration for each technique across different folders
            run_tec_best_conf = recursivedict()
            for run in run_tec_conf_set:
                for tec in run_tec_conf_set[run]:
                    for conf in run_tec_conf_set[run][tec]:
                        if tec not in run_tec_best_conf[run] or self._comparison_operator(run_tec_conf_set[run][tec][conf][self.metric]["hp_selection"], run_tec_best_conf[run][tec][1][self.metric]["hp_selection"]):
                            run_tec_best_conf[run][tec] = (conf, run_tec_conf_set[run][tec][conf])

            # Print results for each run
            for run in range(0, self._campaign_configuration['General']['run_num']):
                unused_techniques = self.techniques
                self._logger.info("Printing results for run %s", run)
                self._logger.info("-->Metrics for all techniques:")
                overall_run_best = ()  # (technique, configuration, metric)

                # Print data of single techniques
                padding = max([len(str(t)) for t in run_tec_best_conf[run]])
                for technique in run_tec_best_conf[run]:
                    temp = run_tec_best_conf[run][technique]
                    if ec.enum_to_configuration_label[technique] in unused_techniques:
                        unused_techniques.remove(ec.enum_to_configuration_label[technique])
                    printed_name = str(technique).ljust(padding)
                    if bool(self._campaign_configuration['General']['details']):
                        for metric, metric_val in temp[1].items():
                            self._logger.info("%s [%s]: (Training %f - HP Selection %f) - Validation %f", printed_name, metric, metric_val["training"], metric_val["hp_selection"], metric_val["validation"])
                    else:
                        metric_val = temp[1][self.metric]
                        self._logger.info("%s [%s]: (Training %f - HP Selection %f) - Validation %f", printed_name, self.metric, metric_val["training"], metric_val["hp_selection"], metric_val["validation"])

                    # Compute which is the best technique
                    if not overall_run_best or self._comparison_operator(temp[1][self.metric]["hp_selection"], overall_run_best[2][self.metric]["hp_selection"]):
                        overall_run_best = (technique, temp[0], temp[1])

                if not overall_run_best:
                    self._logger.error("No valid model was found")
                    exit(1)
                if unused_techniques:
                    self._logger.info("The following techniques had no successful runs: %s", str(unused_techniques))
                self._logger.info("<--Overall best result (according to %s) is %s, with configuration %s", self.metric, overall_run_best[0], overall_run_best[1])
                self._logger.info("Metrics for best result:")
                self._logger.info("-->")
                for metric, metric_val in overall_run_best[2].items():
                    self._logger.info("%s: (Training %f - HP Selection %f) - Validation %f", metric, metric_val["training"], metric_val["hp_selection"], metric_val["validation"])
                self._logger.info("<--")

        elif (validation, hp_selection) in {("KFold", "KFold")}:
            folds = float(self._campaign_configuration['General']['folds'])
            # For each run, for each external fold, for each technique, the aggregated metrics
            run_efold_tec_conf_set = recursivedict()

            # Hyperparameter search aggregating over internal folders
            for conf in self._exp_confs:
                if not conf.trained:
                    continue
                run = int(conf.get_signature()[0].replace("run_", ""))
                ext_fold = int(conf.get_signature()[2].replace("f", ""))
                technique = conf.technique
                configuration = str(conf.get_signature()[4:])
                if self.metric not in run_tec_conf_set[run][technique][configuration]:
                    for metric in conf.metrics:
                        for set_name in set_names:
                            run_tec_conf_set[run][technique][configuration][metric][set_name] = 0
                for metric, metric_vals in conf.metrics.items():
                    for set_name in set_names:
                        run_tec_conf_set[run][technique][configuration][metric][set_name] += (metric_vals[set_name] / (folds * folds))
                if configuration not in run_efold_tec_conf_set[run][ext_fold][technique]:
                    for metric in conf.metrics:
                        for set_name in set_names:
                            run_efold_tec_conf_set[run][ext_fold][technique][configuration][metric][set_name] = 0
                for metric, metric_vals in conf.metrics.items():
                    for set_name in set_names:
                        run_efold_tec_conf_set[run][ext_fold][technique][configuration][metric][set_name] += (metric_vals[set_name] / (folds * folds))

            # Select the best configuration for each technique in each external fold across different internal folders
            run_efold_tec_best_conf = recursivedict()
            for run in run_efold_tec_conf_set:
                for efold in run_efold_tec_conf_set[run]:
                    for tec in run_efold_tec_conf_set[run][efold]:
                        for conf in run_efold_tec_conf_set[run][efold][tec]:
                            if conf not in run_efold_tec_best_conf[run][efold][tec] or self._comparison_operator(run_efold_tec_conf_set[run][efold][tec][conf][self.metric]["hp_selection"], run_efold_tec_best_conf[run][efold][tec][1][self.metric]["hp_selection"]):
                                run_efold_tec_best_conf[run][efold][tec] = (conf, run_efold_tec_conf_set[run][efold][tec][conf], run_efold_tec_conf_set[run][efold][tec][conf])

            # Aggregate on external folds
            run_tec_set = recursivedict()
            for run in run_efold_tec_best_conf:
                for efold in run_efold_tec_best_conf[run]:
                    for tec in run_efold_tec_best_conf[run][efold]:
                        if self.metric not in run_tec_set[run][tec]:
                            for metric in run_efold_tec_best_conf[run][efold][tec][1]:
                                for set_name in set_names:
                                    run_tec_set[run][tec][metric][set_name] = 0
                        for metric in run_efold_tec_best_conf[run][efold][tec][1]:
                            for set_name in set_names:
                                run_tec_set[run][tec][metric][set_name] += run_efold_tec_best_conf[run][efold][tec][1][metric][set_name]

            # Print results for each run
            for run in range(0, self._campaign_configuration['General']['run_num']):
                unused_techniques = self.techniques
                self._logger.info("Printing results for run %s", run)
                self._logger.info("-->Metrics for all techniques:")
                overall_run_best = ()
                # Print data of single techniques
                padding = max([len(str(t)) for t in run_tec_set[run]])
                for technique in run_tec_set[run]:
                    if ec.enum_to_configuration_label[technique] in unused_techniques:
                        unused_techniques.remove(ec.enum_to_configuration_label[technique])
                    printed_name = str(technique).ljust(padding)
                    if bool(self._campaign_configuration['General']['details']):
                        for metric, metric_val in run_tec_set[run][technique].items():
                            self._logger.info("%s [%s]: (Training %f - HP Selection %f) - Validation %f", printed_name, metric, metric_val["training"], metric_val["hp_selection"], metric_val["validation"])
                    else:
                        metric_val = run_tec_set[run][technique][self.metric]
                        self._logger.info("%s [%s]: (Training %f - HP Selection %f) - Validation %f", printed_name, self.metric, metric_val["training"], metric_val["hp_selection"], metric_val["validation"])

                    # Compute which is the best technique
                    if not overall_run_best or self._comparison_operator(run_tec_set[run][technique][self.metric]["hp_selection"], overall_run_best[1][self.metric]["hp_selection"]):
                        overall_run_best = (technique, run_tec_set[run][technique])

                if not overall_run_best:
                    self._logger.error("No valid model was found")
                    exit(1)
                if unused_techniques:
                    self._logger.info("The following techniques had no successful runs: %s", str(unused_techniques))
                self._logger.info("<--Overall best result (according to %s) is %s", self.metric, overall_run_best[0])
                self._logger.info("Metrics for best result:")
                self._logger.info("-->")
                for metric, metric_val in overall_run_best[1].items():
                    self._logger.info("%s: (Training %f - HP Selection %f) - Validation %f", metric, metric_val["training"], metric_val["hp_selection"], metric_val["validation"])
                self._logger.info("<--")

        else:
            self._logger.error("Unexpected combination: %s", str((validation, hp_selection)))
            sys.exit(1)
        best_confs = {}
        best_technique = None
        for conf in self._exp_confs:
            if not conf.trained:
                continue
            technique = conf.technique
            if technique not in best_confs or self._comparison_operator(conf.metrics[self.metric]["validation"], best_confs[technique].metrics[self.metric]["validation"]):
                best_confs[technique] = conf
        for technique in best_confs:
            if not best_technique or self._comparison_operator(best_confs[technique].metrics[self.metric]["validation"], best_confs[best_technique].metrics[self.metric]["validation"]):
                best_technique = technique
        if bool(self._campaign_configuration['General']['details']):
            for run in run_tec_conf_set:
                for tec in run_tec_conf_set[run]:
                    for conf in run_tec_conf_set[run][tec]:
                        print(run_tec_conf_set[run][tec][conf])
                        assert "hp_selection" in run_tec_conf_set[run][tec][conf][self.metric], "hp_selection " + self.metric + " not found for " + str(run) + str(tec) + str(conf)
                        assert "validation" in run_tec_conf_set[run][tec][conf][self.metric], "validation " + self.metric + " not found for " + str(run) + str(tec) + str(conf)
                        self._logger.info("Run %s - Technique %s - Conf %s - Training %s %f - Test %s %f", str(run), ec.enum_to_configuration_label[tec], str(conf), self.metric, run_tec_conf_set[run][tec][conf][self.metric]["hp_selection"], self.metric, run_tec_conf_set[run][tec][conf][self.metric]["validation"])
        return best_confs, best_technique
