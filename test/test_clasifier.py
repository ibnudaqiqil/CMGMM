import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import numpy as np
import os
import pickle   as pkl
import time
from pycm import *
import os
import time
import matplotlib.pyplot as plt
from models.CMGMM import CMGMM
from models.IGMM import IGMM
from sklearn.mixture import GaussianMixture


from models.EvolvingClassifier import EvolvingClassifier
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM
from detector.hddm_a import HDDM_A
from detector.kswin import KSWIN
from detector.kd3 import KD3
#from skmultiflow.drift_detection import HDDM


from sklearn.metrics import accuracy_score

# ## Parameter Setting
import argparse

# help flag provides flag help
# store_true actions stores argument as True

parser = argparse.ArgumentParser()
   
parser.add_argument('-a', '--adaptor', required=False,  default="CMGMM", help="Name of Adaptor {CMGMM/IGMM/GMM}")
parser.add_argument('-d', '--detector', required=False,  default="KD3" , help="Name of Detector {KD3/Adwin/PageHinkley}")
parser.add_argument('-ds', '--dataset', required=False,  default="T1" , help=" Name Of  Dataset to test[T1/T2/T3]")

parser.add_argument('-ws', '--window_size', required=False,  type=int, default=45, help="Parameter for Detector [Windows Length]" )
parser.add_argument('-p1', '--p1', required=False,  type=float, default=0.1, help="Parameter for Detector [Accumulative sum threshold]" )
parser.add_argument('-p2', '--p2', required=False,  type=float, default=0.01, help="Parameter for Detector [Drift Thresehold]" )

parser.add_argument('-p3', '--p3', required=False,  type=int, default=2, help="Parameter for Detector [Drift Thresehold]" )

parser.add_argument('-c', '--cycle', required=False,   default="Number of cycle for passiveLearning" )


args = parser.parse_args() 

DATASET=args.dataset #"T1"
ADAPTOR=args.adaptor #"PageHinkley"
DETECTOR=args.detector#"CMGMM"

df= pd.read_pickle("dataset/exported_800.pickle")
LABEL = df.label.unique()
trainDataset = df[df['status']==1]
testDataset = df[df['status']==2]
df.head()
if DATASET =="T1":
	testDataset_d= pd.read_pickle("dataset/exported_800_t1.pickle")
elif DATASET =="T2":
	testDataset_d= pd.read_pickle("dataset/exported_800_t2.pickle")
elif DATASET =="T3":
	testDataset_d= pd.read_pickle("dataset/exported_800_t3.pickle")


print(testDataset.shape)
print(testDataset_d.shape)

exit()
m0 =testDataset.mfcc.to_numpy()
if DATASET =="T1":
	m1 =testDataset_d.mfcc_t1.to_numpy()
else:
	m1 =testDataset_d.mfcc.to_numpy()
l0 = testDataset.label.to_numpy()
l1 =testDataset_d.label.to_numpy()

TRAIN_DATASET 	= trainDataset.mfcc.to_numpy()
TRAIN_LABEL 	= trainDataset.label.to_numpy()
TEST_DATASET 	= np.append(m0,m1)
TEST_LABEL 		= np.append(l0,l1)

# Incorrect number of clusters
if DETECTOR=="IGMM" :
	adaptor = IGMM(min_components=3, max_components=10)
elif DETECTOR == "CMGMM":
	adaptor = CMGMM(min_components=4, max_components=6)
else:
	adaptor = GaussianMixture(n_components=4)


if DETECTOR == "ADWIN":
	detector = ADWIN()
elif DETECTOR =="PageHinkley":
	detector= PageHinkley()
elif DETECTOR =="HDDM":
	detector= HDDM_A()
elif DETECTOR =="EDDM":
	detector= EDDM()
elif DETECTOR =="KSWIN":
	detector= KSWIN(alpha=0.001,window_size=50, stat_size=15)
elif DETECTOR =="KD3":
	detector= KD3(window_size=args.window_size, 
            accumulative_threshold=args.p2, 
            detection_threshold=args.p1)
else:
	detector = None


model = EvolvingClassifier(adaptor=adaptor,detector=detector,label=LABEL)

model.train(trainDataset,'label','mfcc')
start_time = time.time()
logdrift=[]
if detector != None :
	result,logdrift = model.activeLearning(TEST_DATASET,TEST_LABEL,warningZoneLimit=args.p3)
else:
	CYCLE=int(args.cycle)
	result = model.passiveLearning(TEST_DATASET,TEST_LABEL,CYCLE)
	DETECTOR = DETECTOR+args.cycle

exec_time = (time.time() - start_time)


cm = ConfusionMatrix(actual_vector=TEST_LABEL, predict_vector=result)
mid_= ((cm.overall_stat['95% CI'][0]+cm.overall_stat['95% CI'][1])/2)*100
dist_= (cm.overall_stat['95% CI'][1] - mid_)*100
cm.save_html(os.path.join("report-gs2","Report-{0}-{1}-{2}-{3}-{4}-{5}-{6}".format(DATASET,ADAPTOR,DETECTOR,args.window_size,args.p1,args.p2,args.p3)))
acc=[]
acc_window=[]

for x in range(int(len(result)/100)):
	akhir = (x+1)*100
	window = (x)*100
	
	acc2 = accuracy_score(TEST_LABEL[0:akhir], np.array(result[0:akhir]))
	accw2 = accuracy_score(TEST_LABEL[window:akhir], np.array(result[window:akhir]))
	acc.append(acc2)
	acc_window.append(accw2)

dfx= pd.DataFrame(acc)
dfx['Accuracy Per window'] = acc_window;
dfx.to_excel(os.path.join("report-gs2","Report-{0}-{1}-{2}-{3}-{4}-{5}-{6}.xls".format(DATASET,ADAPTOR,DETECTOR,args.window_size,args.p1,args.p2,args.p3)))

driftx = pd.DataFrame(logdrift)
driftx.to_excel(os.path.join("report-gs2","ReportDrift-{0}-{1}-{2}-{3}-{4}-{5}-{6}.xls".format(DATASET,ADAPTOR,DETECTOR,args.window_size,args.p1,args.p2,args.p3)))

print("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}".format(DATASET,ADAPTOR,DETECTOR,args.window_size,args.p1,args.p2,args.p3,cm.overall_stat['Overall ACC'],cm.overall_stat['F1 Macro'],exec_time,len(logdrift)))




