from skmultiflow.core import ClassifierMixin
from skmultiflow.lazy.base_neighbors import BaseNeighbors
from skmultiflow.utils.utils import *
from sympy import false
from models.CMGMM import CMGMM
from models.Feature import *
from collections import defaultdict
import warnings
import copy

class CMGMMClassifier(BaseNeighbors, ClassifierMixin):
    model= defaultdict()
    def __init__(self,
                 n_neighbors=5,
                 max_window_size=1000,
                 leaf_size=30,
                 metric='euclidean', 
                 classes=[],
                 detector = None,
                 prune_component=False,
                 drift_detector=None
                 
                 ):
        super().__init__(n_neighbors=n_neighbors,
                         max_window_size=max_window_size,
                         leaf_size=leaf_size,
                         metric=metric,)
        self.classes = classes
        self.drift_detector={}
        self.driftData = {}
        self.adaptasi = 0
        self.trained = False
        self.adaptation_log=[]
        for scene_label in self.classes:      
            self.model[scene_label] = CMGMM(min_components=4, max_components=8,pruneComponent=prune_component)
            if(drift_detector != None):
                
                self.drift_detector[scene_label] = copy.deepcopy(drift_detector)
                self.driftData[scene_label]  = []
        if(drift_detector == None):
            self.drift_detector=drift_detector


    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.
            
        y: Array-like
            An array-like containing the classification targets for all 
            samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known classes.

        sample_weight: Not used.
        
        """
        if (self.trained == False):
            data_train = defaultdict()
            for scene_label in self.classes:
                data_train[scene_label] = []
            i = 0
            for dt in X:
                #print(X[i])
                data_train[y[i]].append(X[i])
                i = i+1

            for key in data_train:
                #print(key)
                if (len(data_train[key]) > 0):
                    #print(key,'-',len(data_train[key]))
                    self.model[key].fit(np.array(data_train[key]))
                    #print("n_components => ",key," = ",self.model[key].n_components
            self.trained = True
        elif (self.drift_detector == None):
            data_train =defaultdict()
            for scene_label in self.classes:
                data_train[scene_label]=[]
            i=0
            for dt in X:
                #print(X[i])            
                data_train[y[i]].append(X[i])
                i=i+1
            
            for key in data_train:
                #print(key)
                if (len(data_train[key])>0):
                    #print(key,'-',len(data_train[key]))
                    self.model[key].fit(np.array(data_train[key]))
                    #print("n_components => ",key," = ",self.model[key].n_components) 
        else:
            #train if only detected
            data_train =defaultdict()
            for scene_label in self.classes:
                data_train[scene_label]=[]
            i=0
            
            for dt in X:
                #print(i)            
                label = y[i]
                data_train[y[i]].append(X[i])

                _,_,label_logls = self.predict_detail(X[i],label)
                #print(label_logls)
                self.drift_detector[label].add_element(label_logls)
                self.driftData[y[i]].append(X[i])
                isDetected = self.drift_detector[y[i]].detected_change()
                if(isDetected):
                    #print("adaptation:",label,"=>",i,len(self.driftData[label]))
                    self.adaptasi +=1
                    drifted_data = np.array(self.driftData[label])
                    self.model[y[i]].fit(drifted_data) 
                    self.driftData[y[i]]=[]
                i=i+1

        return self

    def partial_fit_with_lower_bound(self, X, y, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model with lower bound.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.
            
        y: Array-like
            An array-like containing the classification targets for all 
            samples in X.
       
        """
        if (self.trained==False):
            data_train = defaultdict()
            for scene_label in self.classes:
                data_train[scene_label] = []
            i = 0
            for dt in X:
                #print(X[i])
                data_train[y[i]].append(X[i])
                i = i+1

            for key in data_train:
                #print(key)
                if (len(data_train[key]) >0):
                    #print(key,'-',len(data_train[key]))
                    self.model[key].fit(np.array(data_train[key]))
                    #print("n_components => ",key," = ",self.model[key].n_components
            self.trained = True
        elif (self.drift_detector==None):
            data_train =defaultdict()
            for scene_label in self.classes:
                data_train[scene_label]=[]
            i=0
            for dt in X:
                #print(X[i])            
                data_train[y[i]].append(X[i])
                i=i+1
            
            for key in data_train:
                #print(key)
                if (len(data_train[key])>0):
                    #print(key,'-',len(data_train[key]))
                    self.model[key].fit(np.array(data_train[key]))
                    #print("n_components => ",key," = ",self.model[key].n_components) 
        else:
            #train if only detected
            data_train =defaultdict()
            for scene_label in self.classes:
                data_train[scene_label]=[]
            i=0
            
            for dt in X:
                #print(i)            
                label = y[i]
                data_train[y[i]].append(X[i])

                _,_,label_logls = self.predict_detail(X[i],label)
                #print(label_logls)
                self.drift_detector[label].add_element(label_logls)
                self.driftData[y[i]].append(X[i])
                isDetected = self.drift_detector[y[i]].detected_change()
                if(isDetected):
                    print("adaptation:",label,"=>",i,len(self.driftData[label]))
                    self.adaptasi +=1
                    drifted_data = np.array(self.driftData[label])
                    self.model[y[i]].fit(drifted_data) 
                    self.driftData[y[i]]=[]
                i=i+1

        return self
    
    def train(self,data,column_label,column_data):
        for scene_label in self.classes:
            #print ("Train:",scene_label)
            X = np.vstack(data[data[column_label] ==
                               scene_label][column_data].to_numpy())
           
            self.model[scene_label].fit(X)
            #print("n_components => ",scene_label," = ",self.model[scene_label].n_components)  
        self.trained = True
    def predict_detail(self, data,label):
        '''
         Memperediksi satu data mfcc
        '''
        highest_prob=-np.inf
        for scene_label in self.classes:
            #compute likelihood to the labeled model
            logls = self.model[scene_label].score([data])
            #select the highest likelihood as the predicted
            if(label==scene_label):
                label_logls=logls    
            if(highest_prob<logls):
                highest_prob=logls
                predicted_label = scene_label
            
        return predicted_label,highest_prob,label_logls
    
    def _predict(self, data):
        '''
         Memperediksi satu data mfcc
        '''
        highest_prob=-np.inf
        for scene_label in self.classes:
            #compute likelihood to the labeled model
            logls = self.model[scene_label].score([data])
            #select the highest likelihood as the predicted
            if(highest_prob<logls):
                highest_prob=logls
                predicted_label = scene_label
                
        return predicted_label

    def predict_proba(self, X):
        result=[]
        for i in len(X):
            votes = self._predict(X[i])
            result.append(votes)

        return np.asarray(votes)        
    def predict(self, X):
        """ Predict the class label for sample X
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.
            
        Returns
        -------
        numpy.ndarray
            A 1D array of shape (, n_samples), containing the
            predicted class labels for all instances in X.
        
        """
        result=[]
        for x_ in X:
            votes = self._predict(x_)
            result.append(votes)

        return (result)
        
