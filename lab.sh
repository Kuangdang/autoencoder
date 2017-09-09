#block(name=block-1, threads=10, memory=10240, subtasks=1, gpus=1, hours=48)
FIRST=0
IFS=',| '
while read index used mem
do
    if [ "$FIRST" -eq 0 ]; then
        FIRST=1
        continue
    fi
    if [ $used -eq 0 ]; then
        export CUDA_VISIBLE_DEVICES=$index
        break
    fi  
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv)

echo $CUDA_VISIBLE_DEVICES
source activate tensorflow
python $1
