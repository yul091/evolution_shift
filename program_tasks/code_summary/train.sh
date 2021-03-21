#!/bin/bash

RES_DIR='program_tasks/code_summary/result'
if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi

DRST_DIR='data/spring-framework_pkl'
EPOCHS=100
BATCH=512
LR=0.005
EMBEDDING_TYPE=1
EMBEDDING_DIM=100                               # dimension of embedding vectors
EMBEDDING_PATH='/'                              # file for pre-trained vectors
EXPERIMENT_NAME='code_summary'
EXPERIMENT_LOG=$RES_DIR'/'$EXPERIMENT_NAME'.txt'
LAYERS=2                                        # number of rnn layers in the model
MAX_SIZE=10000                                  # number of training samples at each epoch


TK_PATH=$DRST_DIR/tk.pkl
TRAIN_DATA=$DRST_DIR/train.pkl    # file for training dataset
TEST_DATA1=$DRST_DIR/test1.pkl    # file for test dataset1
TEST_DATA2=$DRST_DIR/test2.pkl    # file for test dataset2
TEST_DATA3=$DRST_DIR/test3.pkl    # file for test dataset3
# TEST_DATA4=$DRST_DIR/test4.pkl    # file for test dataset4

# echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=3 python -m program_tasks.code_summary.main \
  --tk_path ${TK_PATH} --epochs ${EPOCHS} --batch ${BATCH} --lr ${LR} \
  --embed_dim ${EMBEDDING_DIM} --embed_path ${EMBEDDING_PATH} \
  --train_data ${TRAIN_DATA} --test_data1 ${TEST_DATA1} --test_data2 ${TEST_DATA2} \
  --test_data3 ${TEST_DATA3} \
  --embed_type ${EMBEDDING_TYPE} --experiment_name ${EXPERIMENT_NAME} \
  --res_dir ${RES_DIR} --layers ${LAYERS} --max_size ${MAX_SIZE} | tee $EXPERIMENT_LOG