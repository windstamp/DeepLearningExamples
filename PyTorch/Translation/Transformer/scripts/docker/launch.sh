#!/bin/bash

# nvcr.io/nvidia/pytorch:20.06-py3 

CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"all"}
DOCKER_BRIDGE=${3:-"host"}

nvidia-docker run -it --name zkl-transformer-torch-fp32-2 \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $PWD:/workspace/translation \
  -v $PWD/results:/results \
  -v $PWD/data:/data \
  transformer_pyt $CMD
