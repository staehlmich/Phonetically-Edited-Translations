#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

import argparse
import typing
from typing import Iterator
from itertools import chain
from collections import Counter
import string
# import Levenshtein
from phoneme_dictionary_extract import PhonDicExtract
import csv
import pandas as pd
from nltk import ngrams

def grapheme_to_phoneme(pronunciation_dic:dict, input_file: str,  output_file:str, concat= False):
    """
    Function that takes a test or training file (1 sentence per line)
    and converts graphemes of tokens into phoneme representation.
    @param pronunciation_dic:
    @param input_file: tokens as grapheme strings
    @param output_file: tokens as phoneme strings
    @param concat: If True, phonemes of tokens are concatenated
    as string without spaces.
    @return:
    """
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.rstrip().split()
            new_line = ""
            # Other option is to use enumerate!
            for i in range(len(line)):
                token = line[i].lower()
                # Write EOS to file.
                if i == len(line)-1:
                    new_line = new_line + token +"\n"
                # Write non EOS tokens to file.
                else:
                    # Write punctuation symbols to file.
                    if token in string.punctuation:
                        new_line = new_line + token + " "
                    # Look up phoneme representation of words and write to file.
                    else:
                        if concat == False:
                            try:
                                new_line = new_line + pronunciation_dic[token]+" "
                                # Special characters
                            except KeyError:
                                new_line = new_line + token + " "
                        else:
                            try:
                                phone_tok = pronunciation_dic[token].replace(" ", "")
                                new_line = new_line + phone_tok+" "
                            #Special characters
                            except KeyError:
                                new_line = new_line + token + " "
            outfile.write(new_line)

#Maybe delete this function. Output not very useful.
def find_similar_types(iter1, iter2):
    similar_types = dict()
    for item1 in iter1:
        #Initiate distance with high number
        min_lev = 2
        for item2 in iter2:
            dist = Levenshtein.distance(item1, item2)
            if dist < min_lev:
                #Saving only strings is enough?
                similar_types[item1] = item2
    return similar_types

def get_homophone_translations(token:str, source:str, translation:str, alignment_file:str) -> Iterator:
    """
    Function that opens a source and reference/target file of aligned
    sentences and prints sentences that contains token on source side.
    @param token: homophone (grapheme strings).
    @param source: Path to source file (grapheme strings).
    @param translation: Path reference/target file (grapheme strings).
    @param alignment_file: Path to fast_align alignment file.
    @return: yields tuples containing: (sent_id, word_id_source, source_token, trans_token, trans_align)
    """
    #STEPS:
    #1. Iterate over files in parallel
    #2. Find homophone string in source.
    #3. Helper function that retrieves alignment for sent_id/word_id --> string to list --> list to dictionary
    #TODO: 4. sliding window --> I want to search/view homophone representations!

    with open(source, "r", encoding="u8") as src, \
         open(translation, "r", encoding="u8") as trans:
        src = src.readlines()
        trans = trans.readlines()
        for i in range(len(src)):
            src_line = src[i]
            if token in src_line:
                for (j, k) in enumerate(src_line.split()):
                    if token.lower() == k.lower():
                            alignment = get_alignment(i, j, alignment_file)
                            try:
                                trans_line = trans[i].split()
                                translation = trans_line[alignment[1]]
                                yield (i+1, j+1, k, translation, alignment[0]+1)
                            #Fast align didn't align anything to token.
                            except TypeError:
                                #Could also be due to poor ASR: I don't want to discard these cases! These interest me!
                                yield (i+1, j+1, k, "NA", 0)

def write_homophone_translations(pdict:dict, source:str, translation:str, alignment_file:str, output:str):
    """
    Function that writes translations of homophones to .csv file.
    @param pdict: dictionary of homophones (as in output of get_multiple_pairings)
    @param source: Path to source file (grapheme strings).
    @param translation: Path reference/target file (grapheme strings).
    @param alignment_file: Path to fast_align alignment file.
    @param output: Path to output .csv file.
    @return:
    """
    with open(output, "w", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, delimiter=" ")
        writer.writerow(["sent_id", "word_id", "source_phon", "source_token", "trans_token", "trans_align"])
        for key in pdict:
            #TODO: sort homophone strings alphabetically (dict keys always change order)
            for value in pdict[key]:
                for elem in get_homophone_translations(value, source, translation, alignment_file):
                    writer.writerow([elem[0], elem[1], key, elem[2], elem[3], elem[4]])

#Could include these functions in phoneme dictionary class.
def return_ngrams(line:str, n:int):
    line = line.translate(str.maketrans('', '', string.punctuation))
    return [ngram for ngram in ngrams(line.rstrip().split(), n)]

### Find homophones & phonetically close strings ###
#Define own class? Or as part of PhonDicExt?
# Upper level --> split phoneme strings into ngrams.
#Careful: Dictionary with phones for correct splitting!
#Careful: new found tokens/phrases have to be in phrase table!

# 1. Simple search: 1 Substring in ngram

# 2. Levenshtein: Find phonetically close tokens. Can't be substrings.
# Different weights: stresses, close location!
# Search for min dist.

# 3. Levenshtein: Find phonetically close tokens in ngram.
# Different weights: stresses, close location!

# 4. Find errors in test data
def find_homophones(hphone_dic:dict, grams: list):
    for gram in grams:
        phone_str = "".join(e for e in gram)
        for key in hphone_dic:
            #Remove whitespaces from dictionary
            joined_key = ''.join(key.split())
            if joined_key not in gram[0]:
                if joined_key not in gram[1]:
                    if joined_key in phone_str:
                        if len(phone_str) > 3:
                            #TODO: phonemes not letters!
                            yield (gram, joined_key, hphone_dic[key])




def main():
    # I want this to compare over word boundaries, but I don't know if this will work.
    # Try to match ngrams and use brevity penalty for longer sentences?
    # Highlight overlaps in phones? How do I filter out garbage?
    # How do I ensure high precision (and recall)?
    # Idea 19.4 --> Train BPE on dictionary and generate learned mappings?

    # 1. Extract phoneme string types (homophone types).
    path_mfa_dic = "/home/user/staehli/master_thesis/homophone_analysis/mfa_output/phrases_dic.mfa"
    phon_ext = PhonDicExtract(filename=path_mfa_dic)
    train_mfa_dic = phon_ext.get_dictionary()
    homophones_en = phon_ext.get_homophone_tuples(train_mfa_dic)
    # train_dataframe_en = pd.DataFrame(homophones_en.items(), columns=["phoneme type", "grapheme type"])
    # train_dataframe_en.to_csv("train.en.freq.csv")

    # 2. Generate phoneme representations by sentence (source side).
    train_phrases = "/home/user/staehli/master_thesis/homophone_analysis/phrases.en"
    grapheme_to_phoneme(train_mfa_dic, train_phrases, "phrases.ph.en", concat=True)

    # 3. Frequencies of homophones by grapheme type in train data (source)
    # train_tc_en = "/home/user/staehli/master_thesis/data/MuST-C/train.tc.en"
    # source_counts = phon_ext.search_homophone_counts(homophones_en, train_tc_en)
    # source_counts = source_counts.groupby(["phone_type", "graph_types", "graph_type"])["graph_type"].count().reset_index(name="count")
    # source_counts.to_csv("src.counts.2.csv", index=False)

    # example = "BAE1K IH0N NUW1 YAO1RK , AY1 AH0M TH HHEH1D AA1F DIH0VEH1LAH0PMAH0NT FAO1R AH0 NAA0N @-@ PRAA1FIH0T KAO1LD RAA1BIH0N HHUH2D ."
    # example_bigrams = return_ngrams(example, 2)
    # find_homophones(source_homophones, example_bigrams)
    with open("test.ph.en", "r", encoding="utf-8") as infile:
        counter = 0
        phon_strings = []
        for line in infile:
            bigrams = return_ngrams(line,2)
            for e in find_homophones(source_homophones, bigrams):
                phon_strings.append(e)
                counter +=1
                print(e)
        print(len(phon_strings))


    #TODO: print out grapheme representations of everything --> lookup with original file.

    #Compare this result against working only with grapheme strings!

    # I should focus on homophone string on the source side. Matching to translations on the target side is hard!


if __name__ == "__main__":
    main()