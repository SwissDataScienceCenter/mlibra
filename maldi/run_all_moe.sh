#!/usr/bin/env sh
NUM_INDUCING_POINTS=$1
LATENT_DIM=$2
FOLD=$3
BATCH_SIZE=$4
NUM_EXPERTS=$5  # New parameter for number of experts
DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
OUTPUT_DIR="/s3/mlibra/mlibra-data/maldi/lgpmoe"  # Changed output directory
MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
N_EPOCHS=40
# check if LEARNING_RATE is set is provided as sixth argument, if not set it to default
if [ -z "$6" ]; then
    LEARNING_RATE=0.001
else
    LEARNING_RATE=$6
fi
SEED=416465
SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/fold_$FOLD.json"
EXP_NAME="CV-FOLD-$FOLD-$LATENT_DIM-$NUM_INDUCING_POINTS-$BATCH_SIZE-MOE-$NUM_EXPERTS"
KERNEL="symmetric"
MODE="lgp"
AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
cd /myhome/mlibra/
pip install -e .
python /myhome/mlibra/maldi/lgpmoe_experiment.py \
    --mode  "lgpmoe" \
    --seed $SEED \
    --num-experts $NUM_EXPERTS \
    --exp-name $EXP_NAME \
    --dataset-path $DATA_PATH \
    --maldi-file $MALDI_FILE \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $N_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --latent-dim $LATENT_DIM \
    --seed $SEED \
    --slices-dataset-file $SLICES_DATASET_FILE \
    --num-inducing $NUM_INDUCING_POINTS \
    --kernel "$KERNEL" \
    --mode "$MODE" \
    --available-lipids-file $AVAILABLE_LIPIDS_FILE \
    --log-transform
