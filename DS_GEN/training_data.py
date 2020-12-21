import os
import dcase_util
import argparse
import pandas as pd
import pickle
import numpy as np
from  MFCC import *

# Setup logging
dcase_util.utils.setup_logging()

log = dcase_util.ui.FancyLogger()
log.title('Extract MFCC Mean Feature')

parser = argparse.ArgumentParser()
parser.add_argument('--datasource', '-d', type=str, default="2019",
                    help='Kode dataset 2016 sd 2019')
parser.add_argument('--target_field', '-t', type=str, default="mfcc",
                    help='Kode dataset 2016 sd 2019')
parser.add_argument('--output', '-o', type=str, default="mfcc",
                    help='Kode dataset 2016 sd 2019')
param = parser.parse_args()

#filename = param.datasource #dataset/exported_800.csv
target_field = param.target_field #dataset/exported_800.csv
#output_filename = param.output #dataset/exported_800.csv
filename = "dataset/exported_800.csv"
output_filename = "exportedwav_800_"+target_field+".pickle"
df= pd.read_csv(filename)


#load MFCC
#mfcc_mean = []
#for index, row in df.iterrows():
#    mfcc_mean.append(extract_feature_mean(row['file']))

#df['mfcc'] = mfcc_mean



#load MFCC
mfcc_t1_mean = []
for index, row in df.iterrows():
    print(index)
    mfcc_t1_mean.append(load_wav(row['file']))

df['mfcc'] = mfcc_t1_mean
df.to_pickle(output_filename)  
