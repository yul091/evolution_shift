#!/bin/bash
# PROJECT=different_project/java_project3
# PROJECT=different_author/elasticsearch
PROJECT=different_time
RES_DIR=program_tasks/code_summary/result/$PROJECT/java_project

if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi

# DATA_DIR=java_data/different_project/java_pkl3
DATA_DIR=java_data/$PROJECT/java_pkl
EPOCHS=300
BATCH=512
LR=0.001
EMBEDDING_TYPE=1
EMBEDDING_DIM=100                               # dimension of embedding vectors
EMBEDDING_PATH='/'                              # file for pre-trained vectors
EXPERIMENT_NAME='code_summary'
EXPERIMENT_LOG=$RES_DIR'/'$EXPERIMENT_NAME'.txt'
LAYERS=2                                        # number of rnn layers in the model
# MAX_SIZE=50000                                # number of training samples at each epoch


TK_PATH=$DATA_DIR/tk.pkl
TRAIN_DATA=$DATA_DIR/train.pkl    # file for training dataset
VAL_DATA=$DATA_DIR/val.pkl        # file for validation dataset
# TEST_DATA=$DATA_DIR/test.pkl      # file for test dataset
TEST_DATA1=$DATA_DIR/test1.pkl    # file for test dataset1
TEST_DATA2=$DATA_DIR/test2.pkl    # file for test dataset2
TEST_DATA3=$DATA_DIR/test3.pkl    # file for test dataset3

# echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=3 python -m program_tasks.code_summary.main \
  --tk_path ${TK_PATH} --epochs ${EPOCHS} --batch ${BATCH} --lr ${LR} \
  --embed_dim ${EMBEDDING_DIM} --embed_path ${EMBEDDING_PATH} \
  --train_data ${TRAIN_DATA} --val_data ${VAL_DATA} \
  --test_data1 ${TEST_DATA1} --test_data2 ${TEST_DATA2} --test_data3 ${TEST_DATA3} \
  --embed_type ${EMBEDDING_TYPE} --experiment_name ${EXPERIMENT_NAME} \
  --res_dir ${RES_DIR} --layers ${LAYERS} | tee $EXPERIMENT_LOG