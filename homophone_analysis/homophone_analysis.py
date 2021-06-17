#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

import argparse
import typing
from typing import Iterator
from itertools import chain
from collections import Counter
import string
import Levenshtein
import phoneme_dictionary_extract as pde
import csv
import pandas as pd
from nltk import ngrams
import json
import timeit

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

def search_homophone_counts(phone_dic, searchfile):
    """
    Method that searches for homophones in test or training data and
    adds them to dataframe.
    @return: df with columns: [phone_type, graph_types, graph_type, sent_id, word_id]
    """
    counts = []
    for phone_type in phone_dic:
        for gtype in phone_dic[phone_type]:
            with open(searchfile, "r", encoding="utf-8") as infile:
                # Initialize counter for sentence id.
                sent_id = 0
                for line in infile:
                    sent_id += 1
                    line = line.rstrip().split()
                    #Use range function to get word_id.
                    for i in range(len(line)):
                        if line[i].lower() == gtype.lower():
                            #convert graph_types to str to search in df.
                            graph_types_str = " ".join(e for e in phone_dic[phone_type])
                            #list with: [phone_type, graph_types,
                            # graph_type, sent_id, word_id]
                            match = [phone_type, graph_types_str, gtype, sent_id, i]
                            counts.append(match)
    return pd.DataFrame(counts, columns=["phone_type", "graph_types", "graph_type", "sent_id", "word_id"])

#Could include these functions in phoneme dictionary class.

def return_ngrams(line:str, n:int):
    """
    Helper function to return ngrams from phonetized phrase table file.
    @param line: Line does not contain punctuation symbols and includes
    phoneme string tags: "<PHON:K<pb>IH1<pb>D>" == "kid"
    @param n: ngram order
    @return:
    """
    #Remove boundary characters "<PHON: ...>"
    line = line[0:-2].replace("<PHON:", "").split("> ")
    #Remove token boundaries <>.
    line = [token[1:-1] for token in line]
    return [ngram for ngram in ngrams(line, n)]

### Find homophones & phonetically close strings ###
#Define own class? Or as part of PhonDicExt?
# Upper level --> split phoneme strings into ngrams.
#Careful: Dictionary with phones for correct splitting!
#Careful: new found tokens/phrases have to be in phrase table!

def join_phon_string(ngram: tuple, mode="phone") -> (str, dict):
    """
    Helper function to join tokens in ngram.
    @param ngram: ngram tuple.
    @param mode: If tokens are in phonetic representation.
    @return:
    """
    #Concatenated tokens of ngram.
    joined_gram = ""
    #Indexes of joined_gram in original tokens.
    mappings = {}
    #Tokens are phonetic strings.
    if mode == "phone":
        joined_gram = "".join((token for token in ngram))
    else:
        joined_gram = "".join((token for token in ngram))
    cum_len = 0
    for n, token in enumerate(ngram):
        #TODO: Error here!
        mappings[tuple(i+cum_len for i in range(0, len(token)))] = n
        cum_len += len(token)
    return joined_gram, mappings

# 1. Simple search: 1 Substring in ngram
def find_phon_substring(search_string, gram: tuple, mode="phone", min_len=6):
    """
    Function that searches for a substring in concatenation of ngrams of
    a line.
    @param search_string: subsrting
    @param grams: List of ngrams from line.
    @param mode: If string is phonemes or graphemes.
    To pass to helper function join_phon_string.
    @param min_len: minimum length of search string.
    Default = longer than a single phoneme.
    @return: if True, return substring.
    """
    # for gram in grams:
    joined_gram, mappings = join_phon_string(gram, mode=mode)
    # print(search_string, joined_gram)
    if len(search_string) > min_len:
        if search_string in joined_gram:
            start_pos = joined_gram.index(search_string)
            end_pos = start_pos+len(search_string)
            # print(start_pos, end_pos)
            start_token = 0
            end_token = 0
            for key in mappings:
                if start_pos in key:
                    start_token = mappings[key]
                if end_pos-1 in key:
                    end_token = mappings[key]
            # print(start_token, end_token)
            if start_token != end_token:
                #Return or yield? Search at line level of ngram level?
                # print(joined_gram, [token for token in gram])
                # print((start_pos, end_pos),(start_token, end_token), mappings)
                return search_string

            #TODO: Restriction: new string has to include stress!

def find_phon_levdist(search_string,phon_string, dist:int):
    """
    Find phonetically close tokens
    @param search_string:
    @param phon_string:
    @param dist:
    @return:
    """
    #Solution by: https://stackoverflow.com/a/19859340
    # def num_there(s):
    #     return any(i.isdigit() for i in s)

    # def stress_in_target(s, stresses):
    #     for stress in stresses:

    stress = [phon+">" for phon in search_string.split(">") if "1" in phon][0]
    if search_string not in phon_string:
        lev_dist = Levenshtein.distance(search_string, phon_string)
        if lev_dist == dist:
            if stress in phon_string:
                return phon_string


# 2. Levenshtein: Find phonetically close tokens. Can't be substrings.
# Different weights: stresses, close location!
# Search for min dist.

# 3. Levenshtein: Find phonetically close tokens in ngram.
# Different weights: stresses, close location!

# ModifiedLev --> class super(Levenshtein))

def main():

    # Idea 19.4 --> Train BPE on dictionary and generate learned mappings?

    # 1. Extract phoneme string types (homophone types).
    path_mfa_dic = "/home/user/staehli/master_thesis/homophone_analysis/mfa_output/phrases_dic.mfa"
    path_full_dic = "/home/user/staehli/master_thesis/homophone_analysis/mfa_output/english.dict"
    phrases_vocab = "/home/user/staehli/master_thesis/homophone_analysis/mfa_input/phrases.vocab.en"
    train_mfa_dic = pde.get_dictionary(path_mfa_dic)
    full_dic = pde.get_dictionary(path_full_dic)
    #Phonetized vocabulary
    vocab_phon = pde.vocab_to_dictionary(phrases_vocab, full_dic, train_mfa_dic)

    #Homophones in train data. TODO: Write this to file, because programm runs slow.
    # counter = 0
    homophones_en = pde.get_homophone_tuples(vocab_phon)
    # print(len(vocab_phon))
    # print(len(homophones_en))
    # for key in homophones_en:
    #     print(key)
    #     if len(homophones_en[key]) > 1:
    #         print(key, homophones_en[key])
    #         counter += len(homophones_en[key])
    # print(counter)

    # 1.b Dictionary as dataframe
    # train_dataframe_en = pd.DataFrame(homophones_en.items(), columns=["phoneme type", "grapheme type"])
    # train_dataframe_en.to_csv("train.en.freq.csv")

    # 2. Generate phoneme representations by sentence and write to file (source side).
    # train_phrases = "/home/user/staehli/master_thesis/homophone_analysis/phrases.en"
    # pde.grapheme_to_phoneme(vocab_phon, train_phrases, "phrases.ph.en")

    ### Find homophones over token boundaries and close phonetic matches. ###

    # print(len(homophones_en))
    # print(list(homophones_en.keys())[0])
    # print(list(homophones_en.keys())[1])
    #
    #TODO: Search function is very slow.
    #TODO: I need to remember the line id it was found in!
    # with open("phrases.ph.short.en", "r") as infile:
    #     for line in infile:
    #         for key in homophones_en:
    #                 ngrams = return_ngrams(line, 2)
                    # for gram in return_ngrams(line, 2):
                    #     match = find_phon_substring(key, gram)
                    #     if match != None:
                    #         print(match, "\n")

    #Test
    # search_string = "<EY1><K>"
    # ngram = ('<W><IY1>', '<T><EY1><K><IH0><NG>>')
    # find_phon_substring(search_string, ngram)

    ex = vocab_phon["glue"]
    for key in homophones_en:
        if find_phon_levdist(ex, key, 4):
            print(key, homophones_en[key])

    # 5. Find errors in test data

    #TODO: print out grapheme representations of everything --> lookup with original file.

    #Compare this result against working only with grapheme strings!

    # I should focus on homophone string on the source side. Matching to translations on the target side is hard!


if __name__ == "__main__":
    main()