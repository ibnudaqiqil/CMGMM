import warnings
warnings.filterwarnings("ignore")
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.mixture.base import _check_X
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection import DDM
from scipy.stats import multivariate_normal as MVN
#import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from numpy.linalg import det

from models.Feature import *
import copy
import logging

class EvolvingClassifier():
	model= defaultdict()
	ph_test=defaultdict()
	drift_detector =defaultdict()
	index=defaultdict()
	
	def __init__(self, 
			strategy="active",#active, passive
			adaptor= None ,
			detector = None,
			augmentation= False,
			label=["a","b"]):
		'''
		Detection Method : [pageHinkley, adwin,eddm,ddm]
		'''
		self.strategy = strategy
		self.setLabel(label)
		self.augmentation=augmentation
		self.setAdaptor(adaptor)
		self.use_detector = False
		if detector is not None:
			self.use_detector = True
			self.setDetector(detector)

	
		self.driftData = {}
		columns = ['label','index','diff', 'diff_sum','status']
		self.driftLog = pd.DataFrame(columns=columns)
	

	def setDetector(self,detector):
		#set Drift Detector
		for scene_label in self.label:	  
			self.drift_detector[scene_label] = copy.deepcopy(detector)   
	def setAdaptor(self,adaptor):
		#set Drift Adaptor
		for scene_label in self.label:
			self.model[scene_label] = copy.deepcopy(adaptor)   
	def setLabel(self,label):
		self.label = label
	
	def printMean(self):
		for scene_label in self.label:
			print("LABEL :",scene_label, "Comp:",self.model[scene_label].n_components)	  
			print("Weight:\n")
			print(self.model[scene_label].weights_)
			print("Mean:\n")
			print(self.model[scene_label].means_)


	def print(self, text):
		return 0
		print(text)

	def resetDriftData(self):
		for scene_label in self.label:
			self.driftData[scene_label]=[]

	def train(self,data,column_label,column_data):
		#Train data for spesific label model
		for scene_label in self.label:
			#print(data[column_label]==scene_label][column_data].to_numpy())
			#raw_freq= (data[data[column_label]==scene_label][column_data].to_numpy())
			mfcc = np.apply_along_axis( extract_mfcc_train, axis=1, arr=data[data[column_label]==scene_label] )
			#print(mfcc)
			self.model[scene_label].fit(mfcc) 		
	


	'''
	def predict(self, data,column_label,column_data):
		predicted=[]
		wrong=0
		
		for index, row in data.iterrows():

			predicted_label,highest_prob = self.score(row[column_data])	
			predicted.append(predicted_label)
			if (self.use_detector):
				self.drift_detector[row['label']].add_element(highest_prob)
				   

			if(row[column_label]!=predicted_label):
				wrong=wrong+1

		
		return predicted
	'''
	def predict(self, data,label):
		'''
		 Memperediksi satu data mfcc
		'''
		highest_prob=-np.inf
		log_ = {}
		for scene_label in self.label:
			#compute likelihood to the labeled model
			logls = self.model[scene_label].score([data])
			log_[scene_label] = logls
			#select the highest likelihood as the predicted
			if(label==scene_label):
				label_logls=logls	
			if(highest_prob<logls):
				highest_prob=logls
				predicted_label = scene_label
			
		return predicted_label,highest_prob,label_logls,log_

	def activeLearning(self,data,label, warningZoneLimit=2):

		predicted=[]
		#wrong={}
		#warningZone=False
		#jumlah_element={}
		warningZoneCount={}
		logDrift=[]
		logDrift2=[]
		loglikelihood=[]
		logs_list=[]
		#reset drift Data
		self.resetDriftData()
		
		key_label_temp =""
		for index  in  range(data.shape[0]):
			#extract The MFCC
			mfcc_row = extract_mfcc(data[index])
			key_label = label[index]

			if(key_label_temp!=key_label):
				logging.debug(f'TRAINING LABEL :{key_label}')
				key_label_temp = key_label
			#do the prediction
			predicted_label,highest_prob,label_logls,logs_ = self.predict(mfcc_row,key_label)
			predicted.append(predicted_label)
			logs_list.append(logs_)
			loglikelihood.append(highest_prob)

			self.drift_detector[key_label].add_element(label_logls)
			isDetected = self.drift_detector[key_label].detected_change()

			#if(warningZone==True):
				#add data along warning zone
			self.driftData[key_label].append(mfcc_row)
			if(self.augmentation):
				self.driftData[key_label].append(augment_TimeStretch(data[index]))
				self.driftData[key_label].append(augment_PitchShift(data[index]))
				self.driftData[key_label].append(augment_Shift(data[index]))
			

			
			

			if(isDetected):
				#print("DETECTED:",key_label," at ",current_counter)
				warningZone=True
				warningZoneCount[key_label] = warningZoneCount.get(key_label, 0) + 1
				if(warningZone):
					
					#do adaptation
					dd= np.array(self.driftData[key_label])
					if(dd.size <50):
						continue

					self.model[key_label].fit(dd)
					logDrift.append(index) 
					logDrift2.append(key_label) 
					
					#flush data and drift marker
					self.driftData[key_label]=[]
					warningZoneCount[key_label]=0
					warningZone=False
					


		
		return predicted, logDrift, logDrift2,loglikelihood,logs_list

	def activeLearningOracle(self,data,label, warningZoneLimit=1,oracle=150):
		print("oracle")
		predicted=[]
		wrong={}
		warningZone=False
		jumlah_element={}
		warningZoneCount={}
		logDrift=[]
		logDrift2=[]
		loglikelihood=[]
		logs_list=[]
		self.resetDriftData()
		key_label_temp =""
		current_counter=1
		oracle_counter={}
		for index  in  range(data.shape[0]):
			row = data[index]
			key_label = label[index]
			if(key_label_temp!=key_label):
				current_counter=1
				key_label_temp = key_label
				logging.debug(f'LABEL :{key_label}')
			#do the prediction
			predicted_label,highest_prob,label_logls,logs_ = self.predict(row,key_label)
			predicted.append(predicted_label)
			logs_list.append(logs_)
			loglikelihood.append(highest_prob)
			#loglikelihood.append(logs_)
			if key_label in oracle_counter:
				oracle_counter[key_label] = oracle_counter[key_label]+1
			else:
				oracle_counter[key_label] = 1

			#push log likelihood to detectpr
			#if (predicted_label==key_label):
			#	print(label_logls)
			#else:
			#	print(">>>>>",label_logls)
			self.drift_detector[key_label].add_element(label_logls)
			isDetected = self.drift_detector[key_label].detected_change()

			#if(warningZone==True):
				#add data along warning zone
			self.driftData[key_label].append(row)
			#self.driftData[key_label].append(row)

			if(isDetected):
				#print("DETECTED:",key_label," at ",current_counter)
				warningZone=True
				warningZoneCount[key_label] = warningZoneCount.get(key_label, 0) + 1
				if(1):
					#print(key_label,":Detectedadapeted@",oracle_counter[key_label]," -- ",oracle_counter[key_label]+250)
				
					#do adaptation
					dd= np.array(self.driftData[key_label])
					if(dd.size <50):
						continue
					self.model[key_label].fit(dd)
					logDrift.append(index) 
					logDrift2.append(key_label) 
					
					#flush data and drift marker
					#self.driftData[key_label]=[]
					warningZoneCount[key_label]=0
					warningZone=False
					oracle_counter[key_label]=1

			if ((oracle_counter[key_label] % oracle ==0) & (oracle_counter[key_label]>0)):
				#print(key_label,"adapeted@",oracle_counter[key_label])
				dd= np.array(self.driftData[key_label])
				self.model[key_label].fit(dd) 
				#flush data and drift marker
				self.driftData[key_label]=[]
				oracle_counter[key_label]=0
				logDrift.append(index) 
				logDrift2.append(key_label) 

			
			current_counter=current_counter+1	

		
		return predicted, logDrift, logDrift2,loglikelihood,logs_list


					
	def passiveLearning(self,data,label,cycle=100):
		predicted=[]
		class_cycle={}
		loglikelihood=[]
		self.resetDriftData()
		for index  in  range(data.shape[0]):
			row = data[index]
			key_label = label[index]
			predicted_label,highest_prob,label_prob,logs_ = self.predict(row,key_label)	
			predicted.append(predicted_label)
			loglikelihood.append(highest_prob)
			self.driftData[key_label].append(row)

			class_cycle[key_label] = class_cycle.get(key_label, 0) + 1
			if (class_cycle[key_label] % cycle==0):
				dd= np.array(self.driftData[key_label])
				self.model[key_label].fit(dd) 
				self.driftData[key_label]=[]
				class_cycle[key_label]=0
		return predicted,loglikelihood
	
	def passiveLearning2(self,data,label,cycle=100):
		predicted=[]
		class_cycle={}
		loglikelihood=[]
		self.resetDriftData()
		for index  in  range(data.shape[0]):
			row = data[index]
			key_label = label[index]
			predicted_label,highest_prob,label_prob,logs_ = self.predict(row,key_label)	
			predicted.append(predicted_label)
			loglikelihood.append(highest_prob)
			self.driftData[key_label].append(row)

			class_cycle[key_label] = class_cycle.get(key_label, 0) + 1
			if (class_cycle[key_label] % cycle==0):
				dd= np.array(self.driftData[key_label])
				self.model[key_label].fit(dd) 
				#self.driftData[key_label]=[]
				class_cycle[key_label]=0
		return predicted,loglikelihood