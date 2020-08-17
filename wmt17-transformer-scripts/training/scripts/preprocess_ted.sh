#!/bin/sh
# Distributed under MIT license

# this sample script preprocesses a sample corpus, including tokenization,
# truecasing, and subword segmentation.
# for application to a different language pair,
# change source and target prefix, optionally the number of BPE operations,

script_dir=`dirname $0`
main_dir=$script_dir/..
data_dir=$main_dir/data
#MS: changed to fit my directories.
dev_dir=$data_dir/dev
test_dir=$data_dir/test
model_dir=$main_dir/model
# variables (toolkits; source and target language)
. $main_dir/vars

#MS: Set by me. Is this correct?
moses_scripts=$main_dir/moses_scripts
bpe_scripts=$main_dir/subword-nmt/subword_nmt
nematus_home=/home/user/staehli/master_thesis/nematus


# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=40000

#minimum number of times we need to have seen a character sequence in the training text before we merge it into one unit
#this is applied to each training text independently, even with joint BPE
bpe_threshold=50

# tokenize
#MS: tokenize training set.
for prefix in corpus
 do
   cat $data_dir/$prefix.$src | \
   $moses_scripts/tokenizer/normalize-punctuation.perl -l $src | \
   $moses_scripts/tokenizer/tokenizer.perl -a -l $src > $data_dir/$prefix.tok.$src

   cat $data_dir/$prefix.$trg | \
   $moses_scripts/tokenizer/normalize-punctuation.perl -l $trg | \
   $moses_scripts/tokenizer/tokenizer.perl -a -l $trg > $data_dir/$prefix.tok.$trg

 done

#MS: tokenize dev set.
for prefix in IWSLT15.TED.dev2010.de-en IWSLT15.TEDX.dev2012.de-en
 do
   cat $dev_dir/$prefix.$src | \
   $moses_scripts/tokenizer/normalize-punctuation.perl -l $src | \
   $moses_scripts/tokenizer/tokenizer.perl -a -l $src > $dev_dir/$prefix.tok.$src

   cat $dev_dir/$prefix.$trg | \
   $moses_scripts/tokenizer/normalize-punctuation.perl -l $trg | \
   $moses_scripts/tokenizer/tokenizer.perl -a -l $trg > $dev_dir/$prefix.tok.$trg

 done

#MS: tokenize test set.
for prefix in IWSLT15.TED.tst2010.de-en.de.de-en IWSLT15.TED.tst2011.de-en.de.de-en IWSLT15.TED.tst2012.de-en.de.de-en IWSLT15.TED.tst2013.de-en.de.de-en IWSLT15.TEDX.tst2013.de-en.de.de-en
 do
   cat $test_dir/$prefix.$src | \
   $moses_scripts/tokenizer/normalize-punctuation.perl -l $src | \
   $moses_scripts/tokenizer/tokenizer.perl -a -l $src > $test_dir/$prefix.tok.$src

   cat $test_dir/$prefix.$trg | \
   $moses_scripts/tokenizer/normalize-punctuation.perl -l $trg | \
   $moses_scripts/tokenizer/tokenizer.perl -a -l $trg > $test_dir/$prefix.tok.$trg

 done

## clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
$moses_scripts/training/clean-corpus-n.perl $data_dir/corpus.tok $src $trg $data_dir/corpus.tok.clean 1 80
##
### train truecaser
$moses_scripts/recaser/train-truecaser.perl -corpus $data_dir/corpus.tok.clean.$src -model $model_dir/truecase-model.$src
$moses_scripts/recaser/train-truecaser.perl -corpus $data_dir/corpus.tok.clean.$trg -model $model_dir/truecase-model.$trg
#
## apply truecaser (cleaned training corpus)
for prefix in corpus
 do
  $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$src < $data_dir/$prefix.tok.clean.$src > $data_dir/$prefix.tc.$src
  $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$trg < $data_dir/$prefix.tok.clean.$trg > $data_dir/$prefix.tc.$trg
 done
#
## apply truecaser (dev/test files)
for prefix in IWSLT15.TED.dev2010.de-en IWSLT15.TEDX.dev2012.de-en
 do
  $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$src < $dev_dir/$prefix.tok.$src > $dev_dir/$prefix.tc.$src
  $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$trg < $dev_dir/$prefix.tok.$trg > $dev_dir/$prefix.tc.$trg
 done
#
#MS: Added test files, since they are in another directory.
for prefix in IWSLT15.TED.tst2010.de-en.de.de-en IWSLT15.TED.tst2011.de-en.de.de-en IWSLT15.TED.tst2012.de-en.de.de-en IWSLT15.TED.tst2013.de-en.de.de-en IWSLT15.TEDX.tst2013.de-en.de.de-en
 do
  $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$src < $test_dir/$prefix.tok.$src > $test_dir/$prefix.tc.$src
  $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$trg < $test_dir/$prefix.tok.$trg > $test_dir/$prefix.tc.$trg
 done
## train BPE
$bpe_scripts/learn_joint_bpe_and_vocab.py -i $data_dir/corpus.tc.$src $data_dir/corpus.tc.$trg --write-vocabulary $data_dir/vocab.$src $data_dir/vocab.$trg -s $bpe_operations -o $model_dir/$src$trg.bpe
#
## apply BPE
#
#MS: First for training data
for prefix in corpus
 do
  $bpe_scripts/apply_bpe.py -c $model_dir/$src$trg.bpe --vocabulary $data_dir/vocab.$src --vocabulary-threshold $bpe_threshold < $data_dir/$prefix.tc.$src > $data_dir/$prefix.bpe.$src
  $bpe_scripts/apply_bpe.py -c $model_dir/$src$trg.bpe --vocabulary $data_dir/vocab.$trg --vocabulary-threshold $bpe_threshold < $data_dir/$prefix.tc.$trg > $data_dir/$prefix.bpe.$trg
 done
##
#MS: Next step is for dev_data
for prefix in IWSLT15.TED.dev2010.de-en IWSLT15.TEDX.dev2012.de-en
 do
  $bpe_scripts/apply_bpe.py -c $model_dir/$src$trg.bpe --vocabulary $data_dir/vocab.$src --vocabulary-threshold $bpe_threshold < $dev_dir/$prefix.tc.$src > $dev_dir/$prefix.bpe.$src
  $bpe_scripts/apply_bpe.py -c $model_dir/$src$trg.bpe --vocabulary $data_dir/vocab.$trg --vocabulary-threshold $bpe_threshold < $dev_dir/$prefix.tc.$trg > $dev_dir/$prefix.bpe.$trg
 done

#MS: Third step is for test_data
for prefix in IWSLT15.TED.tst2010.de-en.de.de-en IWSLT15.TED.tst2011.de-en.de.de-en IWSLT15.TED.tst2012.de-en.de.de-en IWSLT15.TED.tst2013.de-en.de.de-en IWSLT15.TEDX.tst2013.de-en.de.de-en
 do
  $bpe_scripts/apply_bpe.py -c $model_dir/$src$trg.bpe --vocabulary $data_dir/vocab.$src --vocabulary-threshold $bpe_threshold < $test_dir/$prefix.tc.$src > $test_dir/$prefix.bpe.$src
  $bpe_scripts/apply_bpe.py -c $model_dir/$src$trg.bpe --vocabulary $data_dir/vocab.$trg --vocabulary-threshold $bpe_threshold < $test_dir/$prefix.tc.$trg > $test_dir/$prefix.bpe.$trg
 done

## build network dictionaries for separate source / target vocabularies
$nematus_home/data/build_dictionary.py $data_dir/corpus.bpe.$src $data_dir/corpus.bpe.$trg

# build network dictionary for combined source + target vocabulary (for use
# with tied encoder-decoder embeddings)
cat $data_dir/corpus.bpe.$src $data_dir/corpus.bpe.$trg > $data_dir/corpus.bpe.both
$nematus_home/data/build_dictionary.py $data_dir/corpus.bpe.both
