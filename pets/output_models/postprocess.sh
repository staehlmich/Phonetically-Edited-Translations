#! /bin/bash

moses=/home/user/staehli/master_thesis/moses
infile=/home/user/staehli/master_thesis/output/afs_tf_mustc_v2/trans.tok.v2.txt

# post processing
sed -r 's/ \@(\S*?)\@ /\1/g' < $infile |
sed -r 's/\@\@ //g' |
sed "s/&lt;s&gt;//" |
$moses/scripts/tokenizer/detokenizer.perl -l de -no-escape > /home/user/staehli/master_thesis/output/afs_tf_mustc_v2/trans.detok.v2.txt

