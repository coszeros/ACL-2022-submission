#! /bin/bash

NUM_GPUS=$1
shift

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} run_dee_task.py $*
