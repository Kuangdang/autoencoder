#block(name=block-1, threads=10, memory=10240, subtasks=1, gpus=1, hours=24)
    export CUDA_VISIBLE_DEVICES=$1
    echo $CUDA_VISIBLE_DEVICES
    source activate tensorflow
    python ./autoencoder.py
