#!/bin/bash
PROJECT=java
RES_DIR=program_tasks/code_completion/result/$PROJECT

if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi

EPOCHS=300
BATCH=512
LR=0.005
TRAIN_DATA=program_tasks/code_completion/dataset/$PROJECT/train.tsv
VAL_DATA=program_tasks/code_completion/dataset/$PROJECT/val.tsv
TEST_DATA1=program_tasks/code_completion/dataset/$PROJECT/test1.tsv
TEST_DATA2=program_tasks/code_completion/dataset/$PROJECT/test2.tsv
TEST_DATA3=program_tasks/code_completion/dataset/$PROJECT/test3.tsv

EMBEDDING_TYPE=1
EMBEDDING_DIM=100                 #dimension of vectors
EMBEDDING_PATH='/'                #file for pre-trained vectors
EXPERIMENT_NAME='code_completion'
EXPERIMENT_LOG=$RES_DIR'/'$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME

CUDA_VISIBLE_DEVICES=2 python -m program_tasks.code_completion.main \
--train_data=$TRAIN_DATA --val_data=$VAL_DATA \
--test_data1=$TEST_DATA1 --test_data2=$TEST_DATA2 --test_data3=$TEST_DATA3 \
--embedding_type=$EMBEDDING_TYPE --embedding_dim=$EMBEDDING_DIM \
--epochs=$EPOCHS --batch=$BATCH --lr=$LR --res_dir=$RES_DIR \
--embedding_path=$EMBEDDING_PATH \
--experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG

