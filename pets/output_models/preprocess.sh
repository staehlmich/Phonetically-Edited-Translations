#!/bin/bash -v

#path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/home/user/staehli/master_thesis/moses

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/home/user/staehli/master_thesis/subword-nmt
src=/home/user/staehli/master_thesis/data/MuST-C/en-de/data/tst-COMMON/txt/

# tokenize
# should use sacreBLEU for final evaluation

cat $src/tst-COMMON.en \
| $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en \
| $mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -no-escape > test.tok.en