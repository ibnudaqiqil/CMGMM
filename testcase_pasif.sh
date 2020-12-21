#!/bin/bash
 
#Declare a string array
Dataset=( "T1" "T2" "T3









	" )
Detector=( "50" "100" "150" "200"  )
Adaptor=( "IGMM" "EM" "CMGMM")
#Adaptor=( "CMGMM" )

for ds in ${Dataset[*]}; do
     for adp in ${Adaptor[*]}; do
	for dtt in ${Detector[*]}; do
		python test_clasifier.py  --dataset=$ds --adaptor=$adp --detector=PASSIVE --cycle=$dtt
     	done
     done
done

