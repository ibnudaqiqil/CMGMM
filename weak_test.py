import random
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
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection import EDDM
from skmultiflow.drift_detection import HDDM_A
from skmultiflow.drift_detection import HDDM_W
from skmultiflow.drift_detection import KSWIN
from skmultiflow.drift_detection import PageHinkley

from evaluation.WeakPrequential import  WeakEvaluatePrequential
warnings.filterwarnings("ignore")

random.seed(42)

logging.basicConfig(level=logging.WARNING)
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', required=False,
                    default="sea_gen", help="Name of Detector {KD3/Adwin/PageHinkley}")
parser.add_argument('-s', '--label_size', required=False,
                    default=0.25 , help="Name of Detector {KD3/Adwin/PageHinkley}")

args = parser.parse_args()

test_dataset = args.dataset
print("dataset:"+"datasets/"+test_dataset+'.csv')
stream = FileStream("datasets/"+test_dataset+'.csv')
#print(stream.get_target_values())

SGD = SGDClassifier()
eval = WeakEvaluatePrequential(show_plot=True,
                            pretrain_size=1500,
                           batch_size=200,
                           label_size= float(args.label_size),
                           metrics=['accuracy', 'f1', 'running_time', 'model_size'])
#eval.evaluate(stream=stream, model=[SGD], model_names=[ 'SVM-SGD'])
#exit()
detector = KD3(window_size=500)
#detector = ADWIN()
CMMM = CMGMMClassifier(classes=stream.get_target_values(), prune_component=True, drift_detector=None)
SGD = SGDClassifier()
knn_adwin = KNNADWINClassifier(
    n_neighbors=8, leaf_size=40, max_window_size=1000)
SAMKNN = SAMKNNClassifier(n_neighbors=10, weighting='distance', max_window_size=500,
                          stm_size_option='maxACCApprox', use_ltm=False)
learn_pp_nse = LearnPPNSEClassifier()
#CMMM2 = CMGMMClassifier(classes=stream.get_target_values(), prune_component=True, drift_detector=None)
#CMMM.train(train_dataset, 'label', 'mfcc')
#
knn = KNNClassifier(n_neighbors=8, max_window_size=100, leaf_size=40)
eval.evaluate(stream=stream, model=[CMMM], model_names=['CMGMM'])

print(CMMM.adaptasi)
print(eval.psudo_label_accuracy[0])
plt.plot(eval.psudo_label_accuracy[0])

plt.title('model accuracy')
plt.ylabel('Psudo Label accuracy')
plt.xlabel('adaptation')

plt.show(block=True)

print("selesai")


