#!/bin/bash
#cost about 2400s
python3 gen_w2v.py $1 128 10 50 
python3 gen_w2v.py $1 64 20 100 