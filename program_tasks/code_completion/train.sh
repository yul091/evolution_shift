#!/bin/bash

RES_DIR='program_tasks/code_completion/result'
if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi


EPOCHS=50
BATCH=512
LR=0.005
TRAIN_DATA='program_tasks/code_completion/dataset/train.tsv'
TEST_DATA='program_tasks/code_completion/dataset/test.tsv'


EMBEDDING_TYPE=1
EMBEDDING_DIM=100                 #dimension of vectors
EMBEDDING_PATH='/'                #file for pre-trained vectors
EXPERIMENT_NAME='code_completion'
EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=2 python -m program_tasks.code_completion.main \
--train_data=$TRAIN_DATA --test_data=$TEST_DATA --embedding_type=$EMBEDDING_TYPE \
--epochs=$EPOCHS --batch=$BATCH --lr=$LR \
--embedding_dim=$EMBEDDING_DIM --embedding_path=$EMBEDDING_PATH \
--experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG

