#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
echo $CUDA_VISIBLE_DEVICES
source activate tensorflow
echo $(python --version)
python autoencoder.py
