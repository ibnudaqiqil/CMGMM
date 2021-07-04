#!/bin/bash
 
#Declare a string array
Dataset=("exported_800_t1_a1"  "exported_800_t2_a1"  "exported_800_t3_a1")
Detector=("KD3" "ADWIN")
Adaptor=("CMGMM" "IGMM" "EM")
#Adaptor=( "CMGMM" )

for ds in ${Dataset[*]}; do
    for adp in ${Adaptor[*]}; do
	for dtt in ${Detector[*]}; do
		python expereiment_1.py  --dataset=$ds --adaptor=$adp --detector=$dtt
     	done
     done
done

