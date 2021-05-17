import librosa as lr

from glob import glob
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_wav(audio_path, scaling=1, delta=1, chroma=1, mel=1):
    
      return lr.load(audio_path)

def simple_mfcc_load(audio_path):
    y, sr = lr.load(audio_path)
    y_trim = lr.effects.remix(y, intervals=lrdistanc.effects.split(y))
    mfcc = lr.feature.mfcc(y=y_trim, sr=sr)
    return mfcc.T

	
def extract_feature(audio_path, scaling=1, delta=1, chroma=1, mel=1):
    result = np.array([])
    x, sample_rate = lr.load(audio_path,duration=10.0)
    #mfcc
    raw_mfcc=lr.feature.mfcc(x, sr=sample_rate,S=None, n_mfcc=13)
    if(scaling):
        scaler = preprocessing.StandardScaler()
        raw_mfcc  = scaler.fit_transform(raw_mfcc)
   
    if(delta):
        mfcc_delta = lr.feature.delta(raw_mfcc)
        mfcc_delta2 = lr.feature.delta(raw_mfcc, order=2)
        
    feature_array = np.r_[raw_mfcc, mfcc_delta, mfcc_delta2]

    return feature_array


def extract_feature_mean(audio_path, scaling=1, delta=0, chroma=0, mel=0):
    result = np.array([])
    x, sample_rate = lr.load(audio_path,duration=10.0)
    #mfcc
    raw_mfcc	=  lr.feature.mfcc(x, sr=sample_rate)
    if(scaling):
        scaler = preprocessing.StandardScaler()
        raw_mfcc  = scaler.fit_transform(raw_mfcc)
    mfccs =  np.mean(raw_mfcc.T,axis=0)
    result = np.hstack((result, mfccs))
    
    if(delta):
        mfcc_delta = lr.feature.delta(mfccs)
        result = np.hstack((result, mfcc_delta))
        
        mfcc_delta2 = lr.feature.delta(mfccs, order=2)
        result = np.hstack((result, mfcc_delta2))
    
    
    #Compute a chromagram from a waveform or power spectrogram.
    if chroma:
        chromagram =  np.mean(lr.feature.mfcc(y=x, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chromagram))

    #mel
    if mel:
        mels = np.mean(lr.feature.melspectrogram(y=x, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mels))

    return result

def extract_mfcc_mean(audio_path, scaling=0):
    result = np.array([])
    x, sample_rate = lr.load(audio_path,duration=10.0)
    #mfcc
    raw_mfcc	=  lr.feature.mfcc(x, sr=sample_rate)
    if(scaling):
        scaler = preprocessing.StandardScaler()
        raw_mfcc  = scaler.fit_transform(raw_mfcc)
    mfccs =  np.mean(raw_mfcc.T,axis=0)

    return mfccs