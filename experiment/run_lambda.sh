#!/bin/bash
net=$1
expl=$2

sbatch -n 6 --gres=gpu:1 -C TITANRTX -t 70:00:00 job_lambda.sh $net $expl
