from skmultiflow.core import ClassifierMixin
from skmultiflow.lazy.base_neighbors import BaseNeighbors
from skmultiflow.utils.utils import *
from models.CMGMM import CMGMM
from models.Feature import *
from collections import defaultdict
import warnings


class CMGMMClassifier(BaseNeighbors, ClassifierMixin):
    model= defaultdict()
    def __init__(self,
                 n_neighbors=5,
                 max_window_size=1000,
                 leaf_size=30,
                 metric='euclidean', 
                 classes=[],
                 detector = None,
                 ):
        super().__init__(n_neighbors=n_neighbors,
                         max_window_size=max_window_size,
                         leaf_size=leaf_size,
                         metric=metric,)
        self.classes = classes
        for scene_label in self.classes:	  
            self.model[scene_label] = CMGMM(min_components=4, max_components=8,pruneComponent=False)

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
        #print(len(y))
      
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
        return self
    
    def train(self,data,column_label,column_data):
        for scene_label in self.classes:
            #print ("Train:",scene_label)
            self.model[scene_label].fit(np.vstack( data[data[column_label]==scene_label][column_data].head(500).to_numpy())) 	
    
    
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