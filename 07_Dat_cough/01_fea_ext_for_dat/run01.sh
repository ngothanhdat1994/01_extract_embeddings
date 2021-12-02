#!/bin/bash


#python step01_gen_10s.py --dataset 'dev' --outdir '11_10s' --delta 'yes' --dur '10'
#python step01_gen_10s.py --dataset 'test' --outdir '11_10s' --delta 'yes' --dur '10'

python step01_gen_MFCC.py --dataset 'dev' --outdir '11_MFCC' --delta 'yes' --dur '10'
python step01_gen_MFCC.py --dataset 'test' --outdir '11_MFCC' --delta 'yes' --dur '10'



