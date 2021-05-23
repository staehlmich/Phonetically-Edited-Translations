#! /bin/bash

# This example show how to preprocess the audios for MuST-C En-De task


export CUDA_VISIBLE_DEVICES=3
t2t_speech=/home/user/staehli/master_thesis/zero/utils/t2t_speech.py

# we extract 40-dimensional log-Mel filterbanks (melbins)

# handle dev set
#python3 $t2t_speech --melbins 40 --numpad 0 --bs 16 --dataset must_c /home/user/staehli/master_thesis/data/MuST-C/en-de/data/dev dev
# handle test set
#python3 $t2t_speech --melbins 40 --numpad 0 --bs 16 --dataset must_c /home/user/staehli/master_thesis//data/MuST-C/en-de/data/tst-COMMON test
# handle training set
python3 $t2t_speech --melbins 40 --numpad 0 --bs 16 --dataset must_c /home/user/staehli/master_thesis//data/MuST-C/en-de/data/train/ train
