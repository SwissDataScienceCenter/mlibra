#!/usr/bin/env sh5
MEM=50G
CPU=15
GPU=0.5
INDUCINGS=(500 1500)
LATENT_DIMS=(5 20)
BATCH_SIZES=(1000 2000)
FOLDS=(1 2 3 4 5)
for batch_size in  $BATCH_SIZES; do
    for INDUCING in $INDUCINGS; do
      for LATENT_DIM in $LATENT_DIMS; do
        for FOLD in $FOLDS; do
            bash /myhome/mlibra/maldi/run_all.sh "$INDUCING" "$LATENT_DIM" "$FOLD" "$batch_size"
        done
      done
    done
done
