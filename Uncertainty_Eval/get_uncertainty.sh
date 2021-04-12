#!/bin/bash

# MODULE_ID=0 # 0 is code summary
# PROJECT=java_project1
# DATA_DIR=java_data/different_project/java_pkl1
# RES_DIR=program_tasks/code_summary/result/different_project/$PROJECT

MODULE_ID=1 # 1 is code completion
PROJECT=java_project2
DATA_DIR=program_tasks/code_completion/dataset/different_project/$PROJECT
RES_DIR=program_tasks/code_completion/result/different_project/$PROJECT
MAX_SIZE=500


SAVE_DIR=Uncertainty_Results/different_project/$PROJECT

if [ ! -d $SAVE_DIR ]; then
  mkdir $SAVE_DIR
else
  echo dir exist
fi

TRAIN_BATCH_SIZE=64
TEST_BATCH_SIZE=64

CUDA_VISIBLE_DEVICES=6 python -m Metric.test_uncertainty \
--module_id=$MODULE_ID --res_dir=$RES_DIR \
--data_dir=$DATA_DIR --save_dir=$SAVE_DIR \
--train_batch_size=$TRAIN_BATCH_SIZE --test_batch_size=$TEST_BATCH_SIZE \
--max_size=$MAX_SIZE