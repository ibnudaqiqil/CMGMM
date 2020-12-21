from skmultiflow.data import FileStream
from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree
from skmultiflow.trees import HAT
from skmultiflow.drift_detection import ADWIN

from dataset.WaveStream import WaveStream 
from models.CMGMM_Classifier import CMGMMClassifier

from sklearn.linear_model import SGDClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_dataset= pd.read_pickle("dataset/exported_800.pickle")

labels = train_dataset['label'].unique().tolist()
mapping = dict( zip(labels,range(len(labels))) )
train_dataset.replace({'label': mapping},inplace=True)

stream_wave = WaveStream('dataset/exported_800_t1_c1.pickle')
classifier = CMGMMClassifier(classes=stream_wave.get_target_values())
classifier.train(train_dataset,'label','mfcc')
print(stream_wave.get_data_info())
print(stream_wave.get_target_values())
print(stream_wave.get_info())

eval = EvaluatePrequential(show_plot=True,
                           pretrain_size=500,
                           batch_size=200,
                           metrics=['accuracy', 'running_time'])

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