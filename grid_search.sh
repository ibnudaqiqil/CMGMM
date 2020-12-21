#!/bin/bash
WindowSize=("45" "50" "75" )
P1=("0.1" "0.05" "0.01" "0.005" "0.001" "0.0005" "0.0001")
P2=("0.00001")
P3=("2" "3" "4" "5" "6"   )
#Adaptor=( "CMGMM" )

for wsx in ${WindowSize[*]}; do
    for p_1 in ${P1[*]}; do
		for p_3 in ${P3[*]}; do
			echo 'python test_clasifier.py  --adaptor=CMGMM --detector=ADWIN -ws='$wsx' -p1='$p_1' -p2=0.00001 -p3='$p_3
     	done
     done
done

