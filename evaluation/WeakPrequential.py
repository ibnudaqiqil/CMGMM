import sys
import os
import traceback

import warnings
import re
import numpy as np
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score

from numpy import unique

from skmultiflow.evaluation.base_evaluator import StreamEvaluator
from skmultiflow.utils import constants

from sklearn.semi_supervised import LabelSpreading

def averageSmooth(data, windowLen=40):
    return np.convolve(data, np.ones((windowLen,))/windowLen, mode='same')

class WeakEvaluatePrequential(StreamEvaluator):
    """ The prequential evaluation method or interleaved test-then-train method.

    """

    def __init__(self,
                 n_wait=200,
                 max_samples=100000,
                 batch_size=1,
                 pretrain_size=200,
                 max_time=float("inf"),
                 metrics=None,
                 output_file=None,
                 show_plot=False,
                 restart_stream=True,
                 label_size=0.5,
                 data_points_for_classification=False):

        super().__init__()
        self._method = 'weakprequential'
        self.n_wait = n_wait
        self.max_samples = max_samples
        self.pretrain_size = pretrain_size
        self.batch_size = batch_size
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot
        self.data_points_for_classification = data_points_for_classification
        self.psudo_label_accuracy=[]
        
        self.label_size= label_size
        if not self.data_points_for_classification:
            if metrics is None:
                self.metrics = [constants.ACCURACY, constants.KAPPA]

            else:
                if isinstance(metrics, list):
                    self.metrics = metrics
                else:
                    raise ValueError(
                        "Attribute 'metrics' must be 'None' or 'list', passed {}".format(type(metrics)))

        else:
            if metrics is None:
                self.metrics = [constants.DATA_POINTS]

            else:
                if isinstance(metrics, list):
                    self.metrics = metrics
                    self.metrics.append(constants.DATA_POINTS)
                else:
                    raise ValueError(
                        "Attribute 'metrics' must be 'None' or 'list', passed {}".format(type(metrics)))

        self.restart_stream = restart_stream
        self.n_sliding = n_wait

        warnings.filterwarnings(
            "ignore", ".*invalid value encountered in true_divide.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

    def evaluate(self, stream, model, model_names=None):
        """ Evaluates a model or set of models on samples from a stream.

        Parameters
        ----------
        stream: Stream
            The stream from which to draw the samples.

        model: skmultiflow.core.BaseStreamModel or sklearn.base.BaseEstimator or list
            The model or list of models to evaluate.

        model_names: list, optional (Default=None)
            A list with the names of the models.

        Returns
        -------
        StreamModel or list
            The trained model(s).

        """
        self._init_evaluation(model=model, stream=stream,
                              model_names=model_names)

        if self._check_configuration():
            self._reset_globals()
            # Initialize metrics and outputs (plots, log files, ...)
            self._init_metrics()
            self._init_plot()
            self._init_file()

            self.model = self._train_and_test()

            if self.show_plot:
                self.visualizer.hold()

            return self.model

    def _train_and_test(self):
        """ Method to control the prequential evaluation.

        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers.

        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In
        the future, when BaseRegressor is created, it could be an extension from that
        class as well.

        """
        
        self._start_time = timer()
        self._end_time = timer()
        print('Prequential Evaluation')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        actual_max_samples = self.stream.n_remaining_samples()
        if actual_max_samples == -1 or actual_max_samples > self.max_samples:
            actual_max_samples = self.max_samples

        first_run = True
       
        self.psudo_label_accuracy = [[] for _ in range(self.n_models)]
        if self.pretrain_size > 0:
            print('Pre-training on {} sample(s).'.format(self.pretrain_size))

            X, y = self.stream.next_sample(self.pretrain_size)

            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION:
                    # Training time computation
                    self.running_time_measurements[i].compute_training_time_begin(
                    )
                    self.model[i].partial_fit(
                        X=X, y=y, classes=self.stream.target_values)
                    self.running_time_measurements[i].compute_training_time_end(
                    )
                elif self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.running_time_measurements[i].compute_training_time_begin(
                    )
                    self.model[i].partial_fit(
                        X=X, y=y, classes=unique(self.stream.target_values))
                    self.running_time_measurements[i].compute_training_time_end(
                    )
                else:
                    self.running_time_measurements[i].compute_training_time_begin(
                    )
                    self.model[i].partial_fit(X=X, y=y)
                    self.running_time_measurements[i].compute_training_time_end(
                    )
                self.running_time_measurements[i].update_time_measurements(
                    self.pretrain_size)
            self.global_sample_count += self.pretrain_size
            first_run = False

        update_count = 0
        print('Evaluating...')
        ready_to_train=False
        self.pseudo_action = "noisy_label"
        while ((self.global_sample_count < actual_max_samples) & (self._end_time - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
				#only 70%
                self.real_batch_size = int(self.batch_size*self.label_size)
                self.pseudo_batch_size = int(self.batch_size*1-(self.label_size))
 
                #evaluate Real label
                X, y = self.stream.next_sample(self.real_batch_size)


                if X is not None and y is not None:
                    # Test
                    prediction = [[] for _ in range(self.n_models)]
                    for i in range(self.n_models):
                        try:
                            # Testing time
                            self.running_time_measurements[i].compute_testing_time_begin()
                            _hasil_prediksi = self.model[i].predict(X)
                            prediction[i].extend(_hasil_prediksi)
                            self.running_time_measurements[i].compute_testing_time_end()
                        except TypeError:
                            raise TypeError("Unexpected prediction value from {}"
                                            .format(type(self.model[i]).__name__))
                    self.global_sample_count += self.real_batch_size

                    for j in range(self.n_models):
                        for i in range(len(prediction[0])):
                            self.mean_eval_measurements[j].add_result(y[i], prediction[j][i])
                            self.current_eval_measurements[j].add_result(y[i], prediction[j][i])
                    self._check_progress(actual_max_samples)
                    ready_to_train = True

                #Evaluasi Psudo label
                X2, y2 = self.stream.next_sample(self.pseudo_batch_size)
                if X2 is not None and y2 is not None:
                    # variabel penyimpan hasil prediksi
                    psudo_labels = [[] for _ in range(self.n_models)]
                    label_prop_model = LabelSpreading()
                    label_prop_model.fit(X, y)
                    psudoLabel = label_prop_model.predict(X2)
                    #start psudo 
                    '''
                    for i in range(self.n_models):
                        try:
                            # generate the psudolabel
                            self.running_time_measurements[i].compute_testing_time_begin()
                            _hasil_prediksi = self.model[i].predict(X2)
                            psudo_labels[i].extend(_hasil_prediksi)
                            
                            _acc = accuracy_score(y2, _hasil_prediksi)
                            if (not np.isnan(_acc)):
                                self.psudo_label_accuracy[i].append(_acc)
                            
                            self.running_time_measurements[i].compute_testing_time_end()

                        except TypeError:
                            raise TypeError("Unexpected prediction value from {}".format(type(self.model[i]).__name__))
                    '''
                    self.global_sample_count += self.pseudo_batch_size
                    #logging psudolabel accuracy
                  
                    #evaluate the psudo label
                    '''
                    for j in range(self.n_models):
                        for i in range(len(psudo_labels[0])):
                            self.mean_eval_measurements[j].add_result(y2[i], psudo_labels[j][i])
                            self.current_eval_measurements[j].add_result(y2[i], psudo_labels[j][i])
                    '''
                    self._check_progress(actual_max_samples)
                    
                    ready_to_train = True


                    
                if ready_to_train :
                    if first_run:
                        for i in range(self.n_models):
                            if self._task_type != constants.REGRESSION and \
                               self._task_type != constants.MULTI_TARGET_REGRESSION:
                                # Accounts for the moment of training beginning
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y, self.stream.target_values)
                                # Accounts the ending of training
                                self.running_time_measurements[i].compute_training_time_end()
                            else:
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y)
                                self.running_time_measurements[i].compute_training_time_end()

                            # Update total running time
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)
                        first_run = False
                    else:
                        for i in range(self.n_models):
                            self.running_time_measurements[i].compute_training_time_begin()
                            self.model[i].partial_fit(
                                np.concatenate((X, X2)), np.concatenate((y, psudoLabel)))
                            '''S
                            if self.pseudo_action == "noisy_label":
                                self.model[i].partial_weakfit(X2, psudo_labels[i], X, y)
                            elif self.pseudo_action == "noisy_label_verification":
                                self.model[i].partial_weakfit_with_verification(X2, psudo_labels[i], X, y)
                            elif self.pseudo_action == "noisy_label_hedge":
                                self.model[i].partial_weakfit(X2, psudo_labels[i], X, y)
                            '''
                            self.running_time_measurements[i].compute_training_time_end()
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)

                    if ((self.global_sample_count % self.n_wait) == 0 or
                            (self.global_sample_count >= actual_max_samples) or
                            (self.global_sample_count / self.n_wait > update_count + 1)):
                        if psudo_labels is not None:
                            self._update_metrics()
                        update_count += 1
                #end wof



                self._end_time = timer()
             
            except BaseException as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                traceback.print_exc()
                if e is KeyboardInterrupt:
                    self._update_metrics()
                break

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        if len(set(self.metrics).difference({constants.DATA_POINTS})) > 0:
            self.evaluation_summary()
            self.evaluation_summary2()
        else:
            print('Done')

        if self.restart_stream:
            self.stream.restart()

        return self.model


    def evaluation_summary2(self):
        print('Mean performance:')
        for i in range(self.n_models):            
            print('{} {:.4f} {:.4f} {:.3f} {:.3f}'.format(
                self.model_names[i],
                self._data_buffer.get_data(metric_id=constants.ACCURACY, data_id=constants.MEAN)[i],
                self._data_buffer.get_data(metric_id=constants.F1_SCORE, data_id=constants.MEAN)[i],
                self._data_buffer.get_data(metric_id=constants.RUNNING_TIME, data_id='training_time')[i], 
                self._data_buffer.get_data(metric_id=constants.RUNNING_TIME, data_id='testing_time')[i]
            ))
            
           
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fit all the models on the given data.

        Parameters  
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: Array-like
            An array-like containing the classification labels / target values for all samples in X.

        classes: list
            Stores all the classes that may be encountered during the classification task. Not used for regressors.

        sample_weight: Array-like
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
        EvaluatePrequential
            self

        """
        if self.model is not None:
            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION or \
                        self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.model[i].partial_fit(
                        X=X, y=y, classes=classes, sample_weight=sample_weight)
                else:
                    self.model[i].partial_fit(
                        X=X, y=y, sample_weight=sample_weight)
            return self
        else:
            return self

    def predict(self, X):
        """ Predicts with the estimator(s) being evaluated.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        list of numpy.ndarray
            Model(s) predictions

        """
        predictions = None
        if self.model is not None:
            predictions = []
            for i in range(self.n_models):
                predictions.append(self.model[i].predict(X))

        return predictions

    def get_info(self):
        info = self.__repr__()
        if self.output_file is not None:
            _, filename = os.path.split(self.output_file)
            info = re.sub(r"output_file=(.\S+),",
                          "output_file='{}',".format(filename), info)

        return info
