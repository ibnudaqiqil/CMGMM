import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from sklearn.model_selection import train_test_split
from models.CMGMM_Classifier import CMGMMClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(0)

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(y_test)
model = CMGMMClassifier(classes=[0, 1,2])
model.fit(X_train, y_train)
Y_result = model.predict(X_test)
print(Y_result)
print(accuracy_score(y_test, Y_result))
