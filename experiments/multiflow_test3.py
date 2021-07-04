import warnings
warnings.filterwarnings("ignore")

from skmultiflow.data import FileStream
from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree
from skmultiflow.trees import HAT
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection import EDDM
from skmultiflow.drift_detection import HDDM_A
from skmultiflow.drift_detection import HDDM_W
from skmultiflow.drift_detection import KSWIN
from skmultiflow.drift_detection import PageHinkley

import skmultiflow.utils.constants as constants

from detector.kd3 import KD3

from stream.WaveStream import WaveStream 
from stream.MFCCStream import MFCCStream 
from models.CMGMM_Classifier import CMGMMClassifier
from models.IGMM_Classifier import IGMMClassifier

from sklearn.linear_model import SGDClassifier

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg') 

# ## Parameter Setting
import argparse
import logging
import os
from datetime import date

logging.basicConfig( level=logging.WARNING)
# help flag provides flag help
# store_true actions stores argument as True

parser = argparse.ArgumentParser()
   
parser.add_argument('-d', '--dataset', required=False,  default="exported_800_t2.pickle" , help="Name of Detector {KD3/Adwin/PageHinkley}")
parser.add_argument('-p', '--prune', required=False,  default=True , help="Name of Detector {KD3/Adwin/PageHinkley}")

parser.add_argument('-x', '--detector', required=False,  default="NONE" , help="Name of Detector {KD3/Adwin/PageHinkley}")
parser.add_argument('-ws', '--window_size', required=False,  type=int, default=100, help="Parameter for Detector [Windows Length]" )
parser.add_argument('-p1', '--p1', required=False,  type=float, default=0.08, help="Parameter for Detector [Accumulative sum threshold]" )
parser.add_argument('-p2', '--p2', required=False,  type=float, default=0.0001, help="Parameter for Detector [Drift Thresehold]" )
parser.add_argument('-name', '--name', required=False, default="CMGMM", help="Parameter for Detector [Drift Thresehold]" )
parser.add_argument('-bs', '--batch', required=False, type=int, default=100, help="Parameter for Detector [Drift Thresehold]" )

print("Start")
args = parser.parse_args() 

test_dataset=args.dataset 
prune_comp=args.prune 
model_name=args.name 

train_dataset= pd.read_pickle("dataset/mfcc/exported_200.pickle")
train_dataset2= pd.read_pickle("dataset/mfcc/exported_800.pickle")
testDataset = train_dataset2[train_dataset2['status']==2]
train_dataset = train_dataset2[train_dataset2['status']==1]
today = date.today()
result_dir='result/multiflow/'+today.strftime("%Y-%m-%d")+'/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
file_name = "CMGMM-"+model_name+test_dataset+".log"

DETECTOR=args.detector#""
nama_model = "CMGMM"
if (prune_comp):
    nama_model = nama_model+"+ "
else:
    nama_model = nama_model+" "
if DETECTOR == "ADWIN":
    print ("adwin")
    nama_model = nama_model+DETECTOR
    detector = ADWIN()
elif DETECTOR == "DDM":
    print ("DDM")
    nama_model = nama_model+DETECTOR
    detector = DDM()
elif DETECTOR == "EDDM":
    print ("EDDM")
    nama_model = nama_model+DETECTOR
    detector = EDDM()
elif DETECTOR == "HDDM_A":
    print ("HDDM_A")
    nama_model = nama_model+DETECTOR
    detector = HDDM_A()
elif DETECTOR == "HDDM_W":
    print ("HDDM_W")
    nama_model = nama_model+DETECTOR
    detector = HDDM_W()
elif DETECTOR == "KSWIN":
    print ("KSWIN")
    nama_model = nama_model+DETECTOR
    detector = KSWIN()
elif DETECTOR == "PageHinkley":
    print ("PageHinkley")
    nama_model = nama_model+DETECTOR
    detector = PageHinkley()
elif DETECTOR =="KD3":
    nama_model = nama_model+DETECTOR
    detector= KD3(window_size=args.window_size, 
            accumulative_threshold=args.p2, 
            detection_threshold=args.p1,bandwidth=3)
else:
    detector=None

labels = train_dataset['label'].unique().tolist()
mapping = dict( zip(labels,range(len(labels))) )
train_dataset.replace({'label': mapping},inplace=True)

ds = args.dataset 
ds = ds.replace("final_800_", "")
ds = ds.replace("exported_800_", "")
ds = ds.replace(".pickle", "")
#ds = ds.replace("_", " ")
nama_model = nama_model+" ("+ds+")"
stream_wave = MFCCStream('dataset/'+test_dataset,nama_model=nama_model,additional_data= testDataset)

classifier = IGMMClassifier( classes=stream_wave.get_target_values(),prune_component=prune_comp,drift_detector=detector)
classifier.train(train_dataset,'label','mfcc')


eval = EvaluatePrequential(show_plot=True,
                           pretrain_size=0,
                           batch_size=args.batch ,
                           metrics=['accuracy', 'precision','running_time'],
                           output_file=result_dir+file_name,
						   data_points_for_classification=False,
                           n_wait=100 ,

						   )

eval.evaluate(stream=stream_wave, model=classifier, model_names=[model_name])
print(eval._data_buffer.get_data(metric_id=constants.ACCURACY, data_id=constants.MEAN)[0])
print((eval.model[0].adaptasi))
'''

stream = FileStream('dataset/poker.csv')
classifier = SGDClassifier()
eval = EvaluatePrequential(show_plot=True,
                            pretrain_size=500,
                           batch_size=200,
                           metrics=['accuracy', 'kappa', 'running_time', 'model_size'])

eval.evaluate(stream=stream, model=classifier, model_names=['SVM-SGD']);

'''