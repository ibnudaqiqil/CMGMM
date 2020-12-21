import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
import random
import itertools
import bisect
from shutil import copyfile

debug=True
def mydebug(content):
    if(debug):
        print(content)


class conceptSound():
    concept_name =""
    filename =""
    probability = 0.5
    df = None
    def __init__(self, name="",file="",n_number=1, probability=0.5, base_directory="", randomize_file=False):
        self.base_directory = base_directory
       
        self.database = self.base_directory + "/concept.csv"
        self.loadDatase()
        self.concept_name = name
        
        #self.filename = file
        self.filename="/home/abelab/ibunu_i/dcase/dog.wav"
        self.probability = probability
        self.random = randomize_file
        #self.filename = self.getFile(n_number)
        
        
    def loadDatase(self):
        self.df= pd.read_csv(self.database)
    def getFile(self, n):
        #load from dataset
        #concept = self.df[self.df['concept']==self.concept_name]
        if (self.random):
            return self.getConceptRule(random_choise=True,n=1)
        nomer = ('{0:02d}'.format(n))

        return self.base_directory+'/'+self.concept_name+nomer+'.wav'
        #print(concept.iloc[0].concept)
    def getConceptRule(self,random_choise=False, n=1):
        if(random_choise):
            concept_number = int(self.df[self.df['concept']==self.concept_name]['count'])
            n =random.randint(1, concept_number)
            nomer = ('{0:02d}'.format(n))  
            #print(nomer)      
            return self.base_directory+'/'+self.concept_name+nomer+'.wav'
        else:
            return self.getFile(n)


class datasetGenerator():
    target_dir= None
    dataset= None
    concept_rule = None
    rule_name = None
    def __init__(self, target_dir="drifted"):
        self.target_dir = target_dir
        self.lowest_bic = np.infty

    def toss(self,prob):
        tos = random.randrange(0,100)/100
        #print(tos," == ",prob)
        if(tos <=prob):
            return True
        else:
            return False


    def embedSound(self, base_sound,embed_sound, new_sound, times=1, position=0,gain=0):
        base_sound = AudioSegment.from_wav(base_sound)
        
        embed_sound = AudioSegment.from_wav(embed_sound)
        embed_sound = embed_sound.apply_gain(gain)
        final = base_sound.overlay(embed_sound, times=times,position=position)
        final.export(new_sound, format="wav")

    def processSound(self,sound,scene,ds_name,concept_rule,random_choise=False):
        '''
        Proses suara yang akan di embed
        @sound : File suara yang akan di masukkan
        @scene : Nama Scene File tsb

        '''
        print("Proces ",scene,'=>',ds_name," =>")
        #select the rule
        copy= False
        for rule in concept_rule.get(scene).get(ds_name):
            tos = (self.toss(rule.probability))
            target_filename = self.target_dir +"/"+  os.path.basename(sound)
            if(tos):                
                copy=True
                rules = rule.getConceptRule(random_choise=False)
                #print("\t",rule.concept_name,'=>',rules)
                self.embedSound(sound,rules,target_filename,times=random.randrange(1,10),position=random.randrange(0,9000),gain=random.randrange(-20,0))
                sound = target_filename
                
        if (copy == False):
            #just copy the file
            copyfile(sound,target_filename)
        return target_filename

    def loadDataset(self, dataset):
        self.dataset = dataset    
    def loadConceptRule(self, concept_rule):
        self.concept_rule = concept_rule    

    def run(self,base_name, ds_name,rule_name="t1",number_of_concept=2):
        self.rule_name = rule_name

        if(base_name=="base"):
            for index, row in self.dataset.iterrows():
                print(index)
                self.processSound(row["file"],row['label'],ds_name,self.concept_rule)
        else:
            for index, row in self.dataset.iterrows():
                T1_dir='/home/abelab/ibunu_i/dcase/dataset/'+base_name+"/"
                self.processSound(T1_dir+row["filename"],row['label'],ds_name,self.concept_rule)            
