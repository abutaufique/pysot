#!bin/bash
export THONPATH=/home/at7133/Research/pysot
export PYTHONPATH=$PWD:$THONPATH
config=1
python $THONPATH/tools/train.py \
--cfg $config
