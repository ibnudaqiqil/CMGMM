import librosa as lr
from sklearn import preprocessing
import numpy as np


from audiomentations import *

	
def load_wav(audio_path, scaling=1, delta=1, chroma=1, mel=1):
    
      return lr.load(audio_path)

def extract_mfcc(samples,sample_rate=22050, scaling=1):

    #mfcc
    if(type(samples) is tuple):
        samples, sample_rate = samples
    raw_mfcc	=  lr.feature.mfcc(samples, sr=sample_rate)
    if(scaling):
        scaler = preprocessing.StandardScaler()
        raw_mfcc  = scaler.fit_transform(raw_mfcc)
    mfccs =  np.mean(raw_mfcc.T,axis=0)

    return mfccs
def extract_mfcc_train(samples,sample_rate=22050, scaling=0):

    #mfcc
    #print(samples[7])
    if(type(samples[7]) is tuple):
        xsamples, sample_rate = samples[7]
    return extract_mfcc(xsamples,sample_rate, scaling)

def augment_TimeStretch(samples, sample_rate= 16000,min_rate=0.8, max_rate=1.25, p=0.5):
    
    if(type(samples) is tuple):
        samples, sample_rate = samples
    augment = Compose([
        #"""Add gaussian noise to the samples"""
        #AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=min_rate, max_rate=min_rate, p=p),
        #PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        #Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        #AddBackgroundNoise(),
        #TimeMask()
    ])

    # Augment/transform/perturb the audio data
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return extract_mfcc(augmented_samples, sample_rate)

def augment_PitchShift(samples, sample_rate= 16000,min_semitones=-4, max_semitones=4, p=0.5):
    
    if(type(samples) is tuple):
        samples, sample_rate = samples
    augment = Compose([
        #"""Add gaussian noise to the samples"""
        #AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        #TimeStretch(min_rate=min_rate, max_rate=min_rate, p=p),
        PitchShift(min_semitones=-min_semitones, max_semitones=max_semitones, p=p),
        #Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        #AddBackgroundNoise(),
        #TimeMask()
    ])

    # Augment/transform/perturb the audio data
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return extract_mfcc(augmented_samples, sample_rate)

def augment_Shift(samples, sample_rate= 16000,min_fraction=-0.5, max_fraction=0.5, p=0.5):
    
    if(type(samples) is tuple):
        samples, sample_rate = samples
    augment = Compose([
        #"""Add gaussian noise to the samples"""
        #AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        #TimeStretch(min_rate=min_rate, max_rate=min_rate, p=p),
        #PitchShift(min_semitones=-min_semitones, max_semitones=max_semitones, p=p),
        Shift(min_fraction=min_fraction, max_fraction=max_fraction, p=p),
        #AddBackgroundNoise(),
        #TimeMask()
    ])

    # Augment/transform/perturb the audio data
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return extract_mfcc(augmented_samples, sample_rate)