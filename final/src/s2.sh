#!/bin/bash
python3 answer.py $1 64  w2vmodel/dim64win20min1iter100.txt B1 
python3 answer.py $1 128 w2vmodel/dim128win10min1iter50.txt B2 
python3 merge.py answer/B1.csv answer/B2.csv $2