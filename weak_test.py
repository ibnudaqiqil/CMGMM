from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.lazy import KNNClassifier
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.lazy import KNNADWINClassifier
from stream.MFCCStream import MFCCStream
from stream.WaveStream import WaveStream
from datetime import date
import os
import logging
import argparse
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from models.CMGMM_Classifier import CMGMMClassifier
from detector.kd3 import KD3
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import SEAGenerator
from skmultiflow.data import FileStream
import warnings
warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.WARNING)
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', required=False,
                    default="sea", help="Name of Detector {KD3/Adwin/PageHinkley}")

args = parser.parse_args()

test_dataset = args.dataset
print("dataset:"+"datasets/"+test_dataset+'.csv')
stream = FileStream("datasets/"+test_dataset+'.csv')
SGD = SGDClassifier()
knn_adwin = KNNADWINClassifier(
    n_neighbors=8, leaf_size=40, max_window_size=1000)
SAMKNN = SAMKNNClassifier(n_neighbors=10, weighting='distance', max_window_size=500,
                              stm_size_option='maxACCApprox', use_ltm=False)
learn_pp_nse = LearnPPNSEClassifier()

eval = EvaluatePrequential(show_plot=True,
                            pretrain_size=500,
                           batch_size=200,
                           metrics=['accuracy', 'kappa', 'running_time', 'model_size'])
detector = KD3(window_size=50)
CMMM = CMGMMClassifier(classes=stream.get_target_values(), prune_component=True, drift_detector=detector)
#CMMM2 = CMGMMClassifier(classes=stream.get_target_values(), prune_component=True, drift_detector=None)
#CMMM.train(train_dataset, 'label', 'mfcc')
#print(stream.next_sample(100))
knn = KNNClassifier(n_neighbors=8, max_window_size=100, leaf_size=40)
eval.evaluate(stream=stream, model=[
    CMMM,  SGD, learn_pp_nse], model_names=['CMGMM',  'SVM-SGD', 'LNSE+'])
print(CMMM.adaptasi)


