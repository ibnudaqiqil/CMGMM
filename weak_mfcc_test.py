from sklearn.preprocessing import LabelEncoder
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
                    default="final_800_t1_a1.pickle", help="Name of Detector {KD3/Adwin/PageHinkley}")
parser.add_argument('-s', '--label_size', required=False,
                    default=0.25 , help="Name of Detector {KD3/Adwin/PageHinkley}")

args = parser.parse_args()

test_dataset = args.dataset
print("dataset:"+"datasets/"+test_dataset+'.csv')
#ds = ds.replace("_", " ")
nama_model = "WEAK CMGMM ("+test_dataset+")"
train_dataset2 = pd.read_pickle("datasets/mfcc/exported_800.pickle")
labelencoder = LabelEncoder()
train_dataset2['label'] = labelencoder.fit_transform(train_dataset2['label'])
print(train_dataset2.head(10))
testDataset = train_dataset2[train_dataset2['status'] == 2]

stream_wave = MFCCStream('datasets/mfcc/'+test_dataset,
                         nama_model=nama_model, additional_data=testDataset)

eval = WeakEvaluatePrequential(show_plot=True,
                            pretrain_size=0,
                           batch_size=200,
                           label_size= float(args.label_size),
                           metrics=['accuracy', 'f1', 'running_time', 'model_size'])
#eval.evaluate(stream=stream, model=[SGD], model_names=[ 'SVM-SGD'])
#exit()
detector = KD3(window_size=500)
#detector = ADWIN()
print(stream_wave.get_target_values())
CMMM = CMGMMClassifier(classes=stream_wave.get_target_values(),
                            prune_component=True, 
                            drift_detector=None)
CMMM.train(train_dataset2, 'label', 'mfcc')

eval.evaluate(stream=stream_wave, model=[CMMM], model_names=['CMGMM'])

print(CMMM.adaptasi)
print(eval.psudo_label_accuracy[0])
plt.plot(eval.psudo_label_accuracy[0])

plt.title('model accuracy')
plt.ylabel('Psudo Label accuracy')
plt.xlabel('adaptation')

plt.show(block=True)

print("selesai")


