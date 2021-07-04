#!/bin/bash
 
#Declare a string array
Dataset=("exported_800_t1_a1.pickle "  "exported_800_t2_a1.pickle "  "exported_800_t3_a1.pickle " )
#Dataset=("exported_800_t1_a1.pickle "  "exported_800_t2_a1.pickle "  "exported_800_t3_a1.pickle " "exported_800_t1_b1.pickle "  "exported_800_t2_b1.pickle "  "exported_800_t3_b1.pickle " "exported_800_t1_c1.pickle "  "exported_800_t2_c1.pickle "  "exported_800_t3_c1.pickle " "exported_800_t1_c2.pickle "  "exported_800_t2_c2.pickle "  "exported_800_t3_c2.pickle " "exported_800_t1_c3.pickle "  "exported_800_t2_c3.pickle "  "exported_800_t3_c3.pickle " "exported_800_t1_c4.pickle "  "exported_800_t2_c4.pickle "  "exported_800_t3_c4.pickle " "exported_800_t1_c5.pickle "  "exported_800_t2_c5.pickle "  "exported_800_t3_c5.pickle ")

Detector=("KD3")
Adaptor=("CMGMM")
P1=("0.1" "0.05" "0.01" "0.005" "0.001" "0.0005" "0.0001")
P2=("0.00001" "0.0001")
P3=("2" "3" "4" "5" "6" )
#Adaptor=( "CMGMM" )

for ds in ${Dataset[*]}; do
 	for p_1 in ${P1[*]}; do
		for p_2 in ${P2[*]}; do
			python expereiment_2.py  --dataset=$ds --adaptor=CMGMM --detector=KD3 --p1=$p_1 --p2=$p_2
     	done
     done
done

