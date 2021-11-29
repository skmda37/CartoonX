#!/bin/bash

net=$1
expl=$2

cd /home/math/kolek/dev/CartoonX/experiment

python  makedataset.py --dataset=imagenet --num_imgs=100 --model=$net --expl=$expl --expl_params="hparams/$expl.yaml" 
