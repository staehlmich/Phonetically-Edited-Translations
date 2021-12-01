#!/bin/bash

traindir="/home/user/staehli/master_thesis/homophone_analysis/moses_experiments/"
giza_dir="/home/user/staehli/master_thesis/moses/bin/training-tools"
corp=/home"/user/staehli/master_thesis/homophone_analysis/moses_experiments/corpus/train.lc"
train_model="/home/user/staehli/master_thesis/moses/scripts/training"

perl $train_model/train-model.perl -root-dir $traindir -external-bin-dir $giza_dir -mgiza -f de -e en -corpus $corp -last-step 6 -max-phrase-length 5
