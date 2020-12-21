from sklearn.neighbors import KernelDensity
import numpy as np
import sklearn.neighbors.kde as kde
import numpy as np
import scipy.integrate as integrate
import random

import itertools
import math
import collections


class KD3:
    def __init__(self, 
            window_size=45, 
            window_queue_size=10,
            accumulative_threshold=0.00001, 
            detection_threshold=0.0001,
            kernel="gaussian",
            bandwidth=0.5
            ):

        self.kernel =kernel
        self.bandwidth = bandwidth
        self.data_point_sequence = []
        self.data_raw = []

        self.diff_sequence = []  
        self.diff_sum_sequence = []  
        self.drift_detected_seq_nums = [] 

        self.window_size = window_size
        self.accumulative_threshold = accumulative_threshold
        self.accumulative_threshold_to_detect = detection_threshold

        self.current_diff = 0   # may not need to be a memeber
        self.diff_sum = 0



    def calculate_windows_bound(self, window_1, window_2):
        window_1 = window_1+window_2
        #calculate windows bound
        bounds = []
        bounds.append(min(window_1))
        bounds.append(max(window_1))
        return  bounds


    def add_element(self, data_points):
        # data_points must be an iterable
        #print(data_points);
        self.data_point_sequence.extend([data_points])
        #self.data_raw.extend(data_raw)    
       
    def integrate_distance(self, kde1,kde2,bound):
        chk1,e1 = integrate.quad(lambda x: np.exp(kde1.score_samples(np.array([x]).reshape(-1,1)))[0], bound[0], bound[1])
        chk2,e2 = integrate.quad(lambda x: np.exp(kde2.score_samples(np.array([x]).reshape(-1,1)))[0], bound[0], bound[1])
        diff = chk1 - chk2
        # diff = diff * diff
        diff = np.absolute(diff)
        return diff

    def estimate_kde(self,samples,kernel='gaussian',bandwidth=0.5):

        kde_estimator = KernelDensity()
        kde_estimator.fit(samples)
        return kde_estimator



    # Return tuple (is_drift_detected, current_diff, diff_sum)
    def detected_change(self):
   
        sequence_size = len(self.data_point_sequence)
        
        # Not enough data points for two windows
        if (sequence_size < 2 * self.window_size):
            diff = 0
            self.diff_sequence.append(diff)
            self.diff_sum_sequence.append(self.diff_sum)
            return False
        #check wheter data is enough for 1 window??
        if (sequence_size % self.window_size != 0):
            return False

        #return True
        window_2_left_bound = sequence_size - self.window_size
        window_1_left_bound = sequence_size - 2 * self.window_size

        window_1 = self.data_point_sequence[window_1_left_bound: window_2_left_bound]
        window_2 = self.data_point_sequence[window_2_left_bound: ]

        #Calculate the bound form  window 1
        bounds = self.calculate_windows_bound(window_1, window_2)
        window_1 = np.reshape(window_1, (len(window_1),-1))
        window_2 = np.reshape(window_2,(len(window_1),-1))

        kde_window_1 = self.estimate_kde(window_1,self.kernel,self.bandwidth)
        kde_window_2 = self.estimate_kde(window_2,self.kernel,self.bandwidth)
 
    
        diff = self.integrate_distance(kde_window_1, kde_window_2, bounds)
        

       
        if (diff > self.accumulative_threshold):
            self.diff_sum = self.diff_sum + diff

        self.diff_sequence.append(diff)
       

        is_drift_detected = False
        if (self.diff_sum >= self.accumulative_threshold_to_detect):
            # Check whether this happened when diff was rising (rather than falling)
            self.drift_detected_seq_nums.append(sequence_size )
            self.diff_sum = 0
            is_drift_detected = True

        return is_drift_detected


