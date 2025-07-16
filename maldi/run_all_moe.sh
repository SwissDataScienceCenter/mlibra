#!/usr/bin/env sh
NUM_INDUCING_POINTS=$1
LATENT_DIM=$2
FOLD=$3
BATCH_SIZE=$4
NUM_EXPERTS=$5  # New parameter for number of experts
DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
OUTPUT_DIR="/s3/mlibra/mlibra-data/maldi/lgpmoe"  # Changed output directory
MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
N_EPOCHS=20
LEARNING_RATE=0.001
SEED=416465
SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/fold_$FOLD.json"
EXP_NAME="CV-FOLD-$FOLD-$LATENT_DIM-$NUM_INDUCING_POINTS-$BATCH_SIZE-MOE-$NUM_EXPERTS"
KERNEL="symmetric"
MODE="lgp"
AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
cd /myhome/mlibra/
pip install -e .
python /myhome/mlibra/maldi/lgpmoe_experiment.py \
    --config /myhome/mlibra/maldi/configs/$EXP_NAME.json \
    --batch_size $BATCH_SIZE \
    --epochs $N_EPOCHS \
    --lr $LEARNING_RATE \
    --seed $SEED \
    --num_experts $NUM_EXPERTS \
    --device cuda
