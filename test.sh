#!/bin/bash
echo "$(nvidia-smi --query-gpu=index,memory.used --format=csv)"

FIRST=0
ss=5
IFS=',| '
while read index used mem
do
    if [ "$FIRST" -eq 0 ]; then
        FIRST=1
        continue
    fi
    echo $used
    if [ $used -eq 3257 ]; then
        ss=$index
        echo $ss
        break
    fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv)

echo $ss
