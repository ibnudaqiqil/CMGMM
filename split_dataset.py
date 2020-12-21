import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df= pd.read_pickle("dataset/exported_800.pickle")
train, test = train_test_split(df, test_size=0.2)
#print(test)
print(df.groupby(['label']).size())
print(train.groupby(['label']).size())
print(test.groupby(['label']).size())