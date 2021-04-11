#!/bin/bash
PROJECT=different_project/java_project3
RES_DIR=program_tasks/code_completion/result/$PROJECT

if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi

EPOCHS=200
BATCH=512
LR=0.001
TRAIN_DATA=program_tasks/code_completion/dataset/$PROJECT/train.tsv
VAL_DATA=program_tasks/code_completion/dataset/$PROJECT/val.tsv
TEST_DATA=program_tasks/code_completion/dataset/$PROJECT/test.tsv


EMBEDDING_TYPE=1
EMBEDDING_DIM=100                 #dimension of vectors
EMBEDDING_PATH='/'                #file for pre-trained vectors
EXPERIMENT_NAME='code_completion'
EXPERIMENT_LOG=$RES_DIR'/'$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME

CUDA_VISIBLE_DEVICES=2 python -m program_tasks.code_completion.main \
--train_data=$TRAIN_DATA --val_data=$VAL_DATA --test_data=$TEST_DATA \
--embedding_type=$EMBEDDING_TYPE --embedding_dim=$EMBEDDING_DIM \
--epochs=$EPOCHS --batch=$BATCH --lr=$LR --res_dir=$RES_DIR \
--embedding_path=$EMBEDDING_PATH \
--experiment_name=$EXPERIMENT_NAME | tee $EXPERIMENT_LOG

