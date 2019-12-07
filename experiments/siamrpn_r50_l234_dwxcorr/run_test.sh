#!bin/bash
if [ -z "$4" ];then
    echo "Need input parameters"
    echo "Usage: bash `basename "$0"` \$CONFIG \$MODEL \$DATASET \$GPUID"
    exit
fi

export THONPATH=/home/at7133/Research/pysot
export PYTHONPATH=$PWD:$THONPATH
config=$1
model=$2
dataset=$3
gpu=$4
CUDA_VISIBLE_DEVICES=$gpu python $THONPATH/tools/test.py \
--config $config \
--snapshot $model \
--dataset $dataset
