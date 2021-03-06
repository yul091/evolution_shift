#!/bin/bash

## different project
# MODULE_ID=0 # 0 is code summary
# PROJECT=java_project1
# DATA_DIR=java_data/different_project/java_pkl1
# RES_DIR=program_tasks/code_summary/result/different_project/$PROJECT

# MODULE_ID=1 # 1 is code completion
# PROJECT=java_project3
# DATA_DIR=program_tasks/code_completion/dataset/different_project/$PROJECT
# RES_DIR=program_tasks/code_completion/result/different_project/$PROJECT
# MAX_SIZE=200
# TRAIN_BATCH_SIZE=64
# TEST_BATCH_SIZE=64

# SAVE_DIR=Uncertainty_Results/different_project/$PROJECT

# different author
# MODULE_ID=0 # 0 is code summary
# PROJECT=elasticsearch
# DATA_DIR=java_data/different_author/$PROJECT/java_pkl
# RES_DIR=program_tasks/code_summary/result/different_author/$PROJECT
# TRAIN_BATCH_SIZE=256
# TEST_BATCH_SIZE=256

MODULE_ID=1 # 1 is code completion
PROJECT=elasticsearch
DATA_DIR=program_tasks/code_completion/dataset/different_author/$PROJECT
RES_DIR=program_tasks/code_completion/result/different_author/$PROJECT
MAX_SIZE=200
TRAIN_BATCH_SIZE=64
TEST_BATCH_SIZE=64

SAVE_DIR=Uncertainty_Results/different_author/$PROJECT


# # different time
# MODULE_ID=0 # 0 is code summary
# PROJECT=java_project
# DATA_DIR=java_data/different_time/java_pkl
# RES_DIR=program_tasks/code_summary/result/different_time/$PROJECT
# TRAIN_BATCH_SIZE=256
# TEST_BATCH_SIZE=256


# MODULE_ID=1 # 1 is code completion
# PROJECT=java_project
# DATA_DIR=program_tasks/code_completion/dataset/different_time/$PROJECT
# RES_DIR=program_tasks/code_completion/result/different_time/$PROJECT
# TRAIN_BATCH_SIZE=64
# TEST_BATCH_SIZE=64
# MAX_SIZE=200

# SAVE_DIR=Uncertainty_Results/different_time/$PROJECT


if [ ! -d $SAVE_DIR ]; then
  mkdir $SAVE_DIR
else
  echo dir exist
fi



CUDA_VISIBLE_DEVICES=5 python -m Metric.test_uncertainty \
--module_id=$MODULE_ID --res_dir=$RES_DIR \
--data_dir=$DATA_DIR --save_dir=$SAVE_DIR \
--train_batch_size=$TRAIN_BATCH_SIZE --test_batch_size=$TEST_BATCH_SIZE \
--max_size=$MAX_SIZE