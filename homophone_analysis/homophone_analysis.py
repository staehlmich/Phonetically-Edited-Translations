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
    line = line[1:-2].replace("<PHON:", "").split("> ")
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
        joined_gram = "".join((token[1:-1] for token in ngram))
    else:
        joined_gram = "".join((token for token in ngram))
    cum_len = 0
    for n, token in enumerate(ngram):
        #TODO: Error here!
        mappings[tuple(i+cum_len for i in range(0, len(token)))] = n
        cum_len += len(token)
    return joined_gram, mappings

# 1. Simple search: 1 Substring in ngram
def find_phon_substring(search_string, grams: list):
    """
    Function that searches for a substring in concatenation of token in
    ngram.
    @param search_string: subsrting
    @param grams: List of ngrams from line.
    @return: if True, return substring.
    """
    for gram in grams:
        joined_gram, mappings = join_phon_string(gram)
        if search_string in joined_gram:
            start_pos = joined_gram.index(search_string)
            end_pos = start_pos+len(search_string)
            start_token = 0
            end_token = 0
            for key in mappings:
                if start_pos in key:
                    start_token = mappings[key]
                elif end_pos-1 in key:
                    end_token = mappings[key]
            if start_token != end_token:
                return search_string

            #TODO: include phoneme boundaries!
            #TODO: Restriction: new string has to include stress!

            # yield (gram, key, hphone_dic[key])

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

    #Find homophones over token boundaries and close phonetic matches.

    # example = "<PHON:<<DH><IY0>>> <PHON:<<P><UW1><R>>> <PHON:<<K><IH1><D>>>\n"
    # example_bigrams = return_ngrams(example, 2)
    # print(example_bigrams)
    # for gram in example_bigrams:
    #     joined_gram = "".join((token[1:-1] for token in gram))
    #     print(joined_gram)
    # print(phon_string_in_gram("<DH<pb>IY0>", example_bigrams[0]))
    # for elem in find_homophones(homophones_en, example_bigrams):
    #     print(elem)
    # ex2 = [("tail", "or", "made")]
    # print(find_homophones("ilormade", ex2))
    with open("phrases.ph.short.en", "r") as infile:
        for key in homophones_en:
            for line in infile:
                ngrams = return_ngrams(line, 2)
                match = find_phon_substring(key, ngrams)
                if match != None:
                    print(match)

    #Test
    # new_string, mappings = join_phon_string(('<P><ER0><S><EH1><N><T>>', '<AH1><V>'))
    # print(new_string, mappings)




    # 5. Find errors in test data

    #TODO: print out grapheme representations of everything --> lookup with original file.

    #Compare this result against working only with grapheme strings!

    # I should focus on homophone string on the source side. Matching to translations on the target side is hard!


if __name__ == "__main__":
    main()