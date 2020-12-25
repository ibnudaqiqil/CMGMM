import warnings
warnings.filterwarnings("ignore")

from skmultiflow.data import FileStream
from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree
from skmultiflow.trees import HAT
from skmultiflow.drift_detection import ADWIN

from stream.WaveStream import WaveStream 
from stream.MFCCStream import MFCCStream 
from models.CMGMM_Classifier import CMGMMClassifier

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
   
parser.add_argument('-d', '--dataset', required=False,  default="exported_800_t1_a1.pickle" , help="Name of Detector {KD3/Adwin/PageHinkley}")
parser.add_argument('-p', '--prune', required=False,  default=False , help="Name of Detector {KD3/Adwin/PageHinkley}")

args = parser.parse_args() 

test_dataset=args.dataset 
prune_comp=args.dataset 

train_dataset= pd.read_pickle("dataset/mfcc/exported_800.pickle")
today = date.today()
result_dir='result/multiflow/'+today.strftime("%Y-%m-%d")+'/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
file_name = "CMGMM-"+test_dataset+".log"

labels = train_dataset['label'].unique().tolist()
mapping = dict( zip(labels,range(len(labels))) )
train_dataset.replace({'label': mapping},inplace=True)

stream_wave = MFCCStream('dataset/mfcc/'+test_dataset)
classifier = CMGMMClassifier(classes=stream_wave.get_target_values(),prune_component=prune_comp)
classifier.train(train_dataset,'label','mfcc')
print(stream_wave.get_data_info())
print(stream_wave.get_target_values())
print(stream_wave.get_info())

eval = EvaluatePrequential(show_plot=True,
                           pretrain_size=500,
                           batch_size=200,
                           metrics=['accuracy', 'running_time'],
                           output_file=result_dir+file_name,
						   #data_points_for_classification=True

						   )

eval.evaluate(stream=stream_wave, model=classifier, model_names=['CMGMMClassifier'])
'''

stream = FileStream('dataset/poker.csv')
classifier = SGDClassifier()
eval = EvaluatePrequential(show_plot=True,
                            pretrain_size=500,
                           batch_size=200,
                           metrics=['accuracy', 'kappa', 'running_time', 'model_size'])

eval.evaluate(stream=stream, model=classifier, model_names=['SVM-SGD']);

'''