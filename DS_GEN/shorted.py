import warnings
import numpy as np
import pandas as pd


# ## Parameter Setting
import argparse
import logging
import os
from datetime import date

logging.basicConfig( level=logging.WARNING)
# help flag provides flag help
# store_true actions stores argument as True

df= pd.read_pickle("dataset/mfcc/exported_800.pickle")
label_kelas= df['label'].unique()

print(label_kelas)
dfm = pd.DataFrame()
for label in label_kelas:
	dfx =df[df['label']==label].head(200)
	dfm = dfm.append(dfx, ignore_index=True)

print(dfm.shape)
output_filename = "dataset/mfcc/exported_200.pickle"
dfm.to_pickle(output_filename)  

	