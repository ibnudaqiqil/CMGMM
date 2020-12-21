#!/bin/bash
 
#Declare a string array
Dataset=("exported_800_t1_a1.pickle "  "exported_800_t2_a1.pickle "  "exported_800_t3_a1.pickle " "exported_800_t1_b1.pickle "  "exported_800_t2_b1.pickle "  "exported_800_t3_b1.pickle " "exported_800_t1_c1.pickle "  "exported_800_t2_c1.pickle "  "exported_800_t3_c1.pickle " "exported_800_t1_c2.pickle "  "exported_800_t2_c2.pickle "  "exported_800_t3_c2.pickle " "exported_800_t1_c3.pickle "  "exported_800_t2_c3.pickle "  "exported_800_t3_c3.pickle " "exported_800_t1_c4.pickle "  "exported_800_t2_c4.pickle "  "exported_800_t3_c4.pickle " "exported_800_t1_c5.pickle "  "exported_800_t2_c5.pickle "  "exported_800_t3_c5.pickle ")

Detector=("KD3" "ADWIN")
Adaptor=("CMGMM" "IGMM")
#Adaptor=( "CMGMM" )

for ds in ${Dataset[*]}; do
    for adp in ${Adaptor[*]}; do
	for dtt in ${Detector[*]}; do
		python expereiment_2.py  --dataset=$ds --adaptor=$adp --detector=$dtt
     	done
     done
done

