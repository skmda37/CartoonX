#!/bin/bash
net=$1
expl=$2

sbatch -n 6 --gres=gpu:1 -C GF2080Ti -t 70:00:00 job.sh $net $expl
