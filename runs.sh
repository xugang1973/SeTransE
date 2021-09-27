#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

#bash runs.sh {tain|test} {FB15k|WN18}<train_batch_size><embedding_dim><margin><learning_rate><epochs>


#CODE_PATH=codes
#DATA_PATH=data
#SAVE_PATH=models

#The first four parameters must be provided
MODE=$1
DATASET=$2
Train_BATCH_SIZE=${3}
embedding_dim=${4}
margin=${5}
learning_rate=${6}
epochs=${6}

#FULL_DATA_PATH=$DATA_PATH/$DATASET
#SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

#Only used in training
#BATCH_SIZE=$6
#NEGATIVE_SAMPLE_SIZE=$7
#HIDDEN_DIM=$8
#GAMMA=$9
#ALPHA=${10}
#LEARNING_RATE=${11}
#MAX_STEPS=${12}
#TEST_BATCH_SIZE=${13}
#MODULUS_WEIGHT=${14}
#PHASE_WEIGHT=${15}

if [ $MODE == "train" ]
then
echo "Start Training......"
 python -u SeTransE.py --do_train \
            --do_test \
            --dataset $DATASET \
            -tbs $Train_BATCH_SIZE \
            -dim $embedding_dim \
            -mg $margin \
            -lr $learning_rate\
            -ep $epochs\

elif [ $MODE == "test" ]
then
echo "Start Evaluation on Test Data Set......"
 python -u test.py --do_test \
           --dataset $DATASET \

else
   echo "Unknown MODE" $MODE
fi
