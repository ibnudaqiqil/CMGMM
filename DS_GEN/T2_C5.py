import seaborn as sns
from datasetGenerator import *
from MFCC import *
"""
Sudden drift dataset generator

"""

import argparse
import numpy as np
import pandas as pd
import os
from pydub import AudioSegment
parser = argparse.ArgumentParser()

parser.add_argument('--sound_path', help='path of the original sound')
parser.add_argument('--save_path', help='path of the drifted path tobesaved to be saved')


base_dir = "/home/abelab/ibunu_i/dcase/dataset/new_concept"
ruleset= {
    #1
    "airport":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="helicopter",probability=0.7,base_directory=base_dir),  
             conceptSound(name="crowd_bg",probability=0.7,base_directory=base_dir), 
             conceptSound(name="constructionSite_bg",probability=0.7,base_directory=base_dir), 
        ],
        #no overlapping sound, randomized
        "t2":[
             conceptSound(name="constructionSite_bg",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="crowd_bg",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="helicopter",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],
        #overlapping, not random
        "t3":[
             conceptSound(name="constructionSite_bg",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="crowd_bg",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="helicopter",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="plane",probability=0.7,base_directory=base_dir,randomize_file=True),
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="children_playing",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],       
        #overlapping, random
        
        #large overlapping, random
               
    },
    #2
    "beach":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="Person_swim",probability=0.7,base_directory=base_dir),  
             conceptSound(name="foot_step_sand",probability=0.7,base_directory=base_dir), 
             conceptSound(name="rain_bg",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="Person_swim",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="foot_step_sand",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="rain_bg",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],     
        "t3":[
             conceptSound(name="Person_swim",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="voice",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="rain_bg",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Teenage_crowd",probability=0.7,base_directory=base_dir), 
             conceptSound(name="dog",probability=0.7,base_directory=base_dir), 
             
        ],     
    },
    #3
    "bus":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="carHorn",probability=0.7,base_directory=base_dir),  
             conceptSound(name="engine",probability=0.7,base_directory=base_dir), 
             conceptSound(name="cityCar",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="carHorn",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="engine",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="cityCar",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],
        "t3":[
             conceptSound(name="carHorn",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="engine",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="cityCar",probability=0.7,base_directory=base_dir,randomize_file=True),
             conceptSound(name="voice",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="ambulance",probability=0.7,base_directory=base_dir,randomize_file=True),
             conceptSound(name="Teenage_crowd",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],        
      
    },
    #4
    "city_center":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="bird_bg",probability=0.7,base_directory=base_dir),  
             conceptSound(name="constructionSite_bg",probability=0.7,base_directory=base_dir), 
             conceptSound(name="ambulance",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="bird_bg",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="constructionSite_bg",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="ambulance",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],
        "t3":[
             conceptSound(name="bird_bg",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="constructionSite_bg",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="ambulance",probability=0.7,base_directory=base_dir,randomize_file=True),
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="children_playing",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="phone_ringing",probability=0.7,base_directory=base_dir),
        ],       
    },
    #5
    "grocery_store":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir),  
             conceptSound(name="children_playing",probability=0.7,base_directory=base_dir), 
             conceptSound(name="ShoppingCart",probability=0.7,base_directory=base_dir), 
        ],
         "t2":[
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="children_playing",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="ShoppingCart",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],
        "t3":[
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="children_playing",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="ShoppingCart",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Vacum_cleaner",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="phone_ringing",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],      
    },
    #6
    "home":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="frying",probability=0.7,base_directory=base_dir),  
             conceptSound(name="doorHouse",probability=0.7,base_directory=base_dir), 
             conceptSound(name="Vacum_cleaner",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="frying",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="doorHouse",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Vacum_cleaner",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],
        "t3":[
             conceptSound(name="frying",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="doorHouse",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Vacum_cleaner",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Clock",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="phone_ringing",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Washing_machine",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],
             
    },
    #7
    "metro":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="siren",probability=0.7,base_directory=base_dir),  
             conceptSound(name="roadCar",probability=0.7,base_directory=base_dir), 
             conceptSound(name="thunder",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="siren",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="roadCar",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="thunder",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],
       "t3":[
             conceptSound(name="siren",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="roadCar",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="thunder",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="crowd_bg",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Wind",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],
      
    },
    #8
    "office":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="typing",probability=0.7,base_directory=base_dir),  
             conceptSound(name="phone_ringing",probability=0.7,base_directory=base_dir), 
             conceptSound(name="sneeze",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="typing",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="phone_ringing",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="sneeze",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],
        "t3":[
             conceptSound(name="typing",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="phone_ringing",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="sneeze",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="broom",probability=0.7,base_directory=base_dir),
             conceptSound(name="Camera",probability=0.7,base_directory=base_dir), 
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir,randomize_file=True),
        ],
       
    },
    #9
    "public_square":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir),  
             conceptSound(name="broom",probability=0.7,base_directory=base_dir), 
             conceptSound(name="plane",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="broom",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="plane",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],
         "t3":[
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="broom",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="plane",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="bird_bg",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="rain_bg",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Teenage_crowd",probability=0.7,base_directory=base_dir),  
             
        ],      
    },
    #10 residential_area
    "residential_area":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="Wind",probability=0.7,base_directory=base_dir),  
             conceptSound(name="Camera",probability=0.7,base_directory=base_dir), 
             conceptSound(name="Washing_machine",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="Wind",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="Camera",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Washing_machine",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],
        "t3":[
             conceptSound(name="Wind",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="Camera",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Washing_machine",probability=0.7,base_directory=base_dir,randomize_file=True), 
              conceptSound(name="bird_bg",probability=0.7,base_directory=base_dir,randomize_file=True),
              conceptSound(name="sneeze",probability=0.7,base_directory=base_dir,randomize_file=True), 
              conceptSound(name="Clock",probability=0.7,base_directory=base_dir,randomize_file=True),
        ],
       
    },
    #11 shopping_mall
    "shopping_mall":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="Clock",probability=0.7,base_directory=base_dir),  
             conceptSound(name="Camera",probability=0.7,base_directory=base_dir), 
             conceptSound(name="Teenage_crowd",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="Clock",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="Camera",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Teenage_crowd",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],
         "t3":[
             conceptSound(name="Clock",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="Camera",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Teenage_crowd",probability=0.7,base_directory=base_dir,randomize_file=True), 
                          conceptSound(name="children_playing",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="phone_ringing",probability=0.7,base_directory=base_dir),
             conceptSound(name="dog",probability=0.7,base_directory=base_dir),
        ],
       
    },
    #12 street_pedestrian
    "street_pedestrian":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="dog",probability=0.7,base_directory=base_dir),  
             conceptSound(name="Bicyle",probability=0.7,base_directory=base_dir), 
             conceptSound(name="broom",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="dog",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="Bicyle",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="broom",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ], 
        "t3":[
             conceptSound(name="dog",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="Bicyle",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="broom",probability=0.7,base_directory=base_dir,randomize_file=True), 
              conceptSound(name="foot_step",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="children_playing",probability=0.7,base_directory=base_dir,randomize_file=True), 
             
        ],       
          
    },    
    #13 street_traffic
    "street_traffic":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="Motor_car_passing",probability=0.7,base_directory=base_dir),  
             conceptSound(name="dog",probability=0.7,base_directory=base_dir), 
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="Motor_car_passing",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="dog",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],       
          "t3":[
             conceptSound(name="Motor_car_passing",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="dog",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="foot_step",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="siren",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="plane",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="bell",probability=0.7,base_directory=base_dir,randomize_file=True),  
             
             
        ],       
    },
    #14 tram
    "tram":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="coughing",probability=0.7,base_directory=base_dir),  
             conceptSound(name="bell",probability=0.7,base_directory=base_dir), 
             conceptSound(name="Footstep_pavement",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="coughing",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="bell",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Footstep_pavement",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],      

                "t3":[
             conceptSound(name="coughing",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="bell",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="Footstep_pavement",probability=0.7,base_directory=base_dir,randomize_file=True), 
              conceptSound(name="Teenage_crowd",probability=0.7,base_directory=base_dir,randomize_file=True), 
                          conceptSound(name="children_playing",probability=0.7,base_directory=base_dir,randomize_file=True), 
           
        ],    
    },
    #15 cafe/restaurant
    "cafe/restaurant":{
        #no overlaping sound (distinct sound each scene)
        "t1":[
             conceptSound(name="Washing_machine",probability=0.7,base_directory=base_dir),  
             conceptSound(name="Food_mixer",probability=0.7,base_directory=base_dir), 
             conceptSound(name="kitchenware",probability=0.7,base_directory=base_dir), 
        ],
        "t2":[
             conceptSound(name="Washing_machine",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="Food_mixer",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="kitchenware",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],  
         "t3":[
             conceptSound(name="Washing_machine",probability=0.7,base_directory=base_dir,randomize_file=True),  
             conceptSound(name="Food_mixer",probability=0.7,base_directory=base_dir,randomize_file=True), 
             conceptSound(name="kitchenware",probability=0.7,base_directory=base_dir,randomize_file=True), 
              conceptSound(name="Teenage_crowd",probability=0.7,base_directory=base_dir,randomize_file=True), 
                          conceptSound(name="children_playing",probability=0.7,base_directory=base_dir,randomize_file=True), 
        ],       
    },
    

}



#dataset_gen.processSound("dataset_gen","bus",rule_t1)
base_path = "/home/abelab/ibunu_i/R1/"
filename = base_path+"dataset/exported_800.csv"
TX ="t2"
target_t=TX+"_c5"
#read csv
df= pd.read_csv(filename)

dirpath = os.getcwd()

target = os.path.join(dirpath, "dataset/"+target_t)
print("====test schema :",target_t)
print("====Create dir :",target)
try:
     os.mkdir(target)
except OSError:
    pass

dataset_gen = datasetGenerator(target_dir=target)
dataset_gen.loadDataset(df)
dataset_gen.loadConceptRule(ruleset)
#dataset_gen.rule_name = rule_name
label_kelas= df['label'].unique()

#print (label_kelas)
mfcc_list = []
gain_list= []
position_list= []
times_list= []
concept_name_list= []
from shutil import copyfile
for label in label_kelas:

#def prosesSceneSound(datasetSound,label,rule,dataset_gen):
    #prosesSceneSound(df[df['label']==l],l,rule_t1,dataset_gen)
    for index, row in df[df['label']==label].iterrows():

        dirpath = os.getcwd()
        target_dir = os.path.join(dirpath, "dataset/"+target_t)
        times=random.randrange(1,10)
        position=random.randrange(0,9000)
        gain = random.randrange(-20,0)
        target_filename =target_dir +"/"+  os.path.basename(row['file'])

 


        if (index <= 100):
            #print()
            rules = ""
            concept_name =""
            #dataset_gen.embedSound(row['file'],rules,target_filename,times=times,position=position,gain=gain)
            
            copyfile(row['file'], target_filename)

            
        elif (index <= 150):

            rules = ruleset[label][TX][0].getFile(1)
            concept_name =ruleset[label][TX][0].concept_name
            dataset_gen.embedSound(row['file'],rules,target_filename,times=times,position=position,gain=gain)

        elif (index <= 250):
            rules = ruleset[label][TX][0].getFile(1)
            concept_name =ruleset[label][TX][0].concept_name
            dataset_gen.embedSound(row['file'],rules,target_filename,times=times,position=position,gain=gain)
            rules = ruleset[label][TX][1].getFile(1)
            concept_name =ruleset[label][TX][1].concept_name
            dataset_gen.embedSound(target_filename,rules,target_filename,times=times,position=position,gain=gain)
            
        elif (index <= 300):
            rules = ruleset[label][TX][0].getFile(1)
            concept_name =ruleset[label][TX][0].concept_name
            dataset_gen.embedSound(row['file'],rules,target_filename,times=times,position=position,gain=gain)

        elif (index <= 500):
            #print()
            rules = ""
            concept_name =""
            #dataset_gen.embedSound(row['file'],rules,target_filename,times=times,position=position,gain=gain)
            
            copyfile(row['file'], target_filename)
            
        elif (index <= 550):

            rules = ruleset[label][TX][0].getFile(1)
            concept_name =ruleset[label][TX][0].concept_name
            dataset_gen.embedSound(row['file'],rules,target_filename,times=times,position=position,gain=gain)

        elif (index <= 650):
            rules = ruleset[label][TX][0].getFile(1)
            concept_name =ruleset[label][TX][0].concept_name
            dataset_gen.embedSound(row['file'],rules,target_filename,times=times,position=position,gain=gain)
            rules = ruleset[label][TX][2].getFile(1)
            concept_name =ruleset[label][TX][2].concept_name
            dataset_gen.embedSound(target_filename,rules,target_filename,times=times,position=position,gain=gain)
            
        elif (index <= 700):
            rules = ruleset[label][TX][0].getFile(1)
            concept_name =ruleset[label][TX][0].concept_name
            dataset_gen.embedSound(row['file'],rules,target_filename,times=times,position=position,gain=gain)
            
        else:
            #print()
            rules = ""
            concept_name =""
            #dataset_gen.embedSound(row['file'],rules,target_filename,times=times,position=position,gain=gain)
            
            copyfile(row['file'], target_filename)


        #extract mfcc
        print("Extract mfcc: ",target_t,">",concept_name,">>",target_filename)
        mfcc_list.append(extract_feature_mean(target_filename))
        
        gain_list.append(gain)
        position_list.append(position)
        times_list.append(times)
        concept_name_list.append(concept_name)


df['mfcc'] = mfcc_list
df['gain'] = gain_list
df['position'] = position_list
df['times'] = times_list
df['new_concept_name'] = concept_name_list

output_filename = "exported_800_"+target_t+".pickle"
df.to_pickle(output_filename)  