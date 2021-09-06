#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

import argparse
import itertools
import pstats
import typing
from typing import Iterator
from itertools import chain
from collections import Counter
import re
import string
import Levenshtein
import csv
import pandas as pd
from nltk import ngrams
from nltk.metrics import aline
import json
import timeit
import linecache
import numpy as np
import cProfile
import Levenshtein

#Own code
import phoneme_dictionary_extract as pde
import iterative_sentence_splitting as its

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

def minimum_edit_distance(source: str, target: str):
    """
    Returns the minimum edit distance between two strings @param source and
    @param target, given the cost of
        insertion    = 1;
        deletion     = 1;
        substitution = 1.
    Original function by: Samuel Läubli 2019. Modified by: Michael Stähli.
    """
    #Keep phoneme boundaries.
    source = split_phonemes(source)
    target = split_phonemes(target)
    n = len(source)
    m = len(target)
    d = [[0 for _ in range(m+1)] for _ in range(n+1)]
    d[0][0] = 0
    for i in range(1, n+1):
        d[i][0] = d[i-1][0] + 1
    for j in range(1, m+1):
        d[0][j] = d[0][j-1] + 1
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i][j] = min(
                d[i-1][j] + 1, # del
                d[i][j-1] + 1, # ins
                d[i-1][j-1] + (1 if source[i-1] != target[j-1] else 0) # sub
                )
    return d[n][m]



def find_phon_levdist(search_string:list, phon_string: list):
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

    #TODO: Probably can do this more efficiently. Plus, How do I handle numbers?
    stress = [phon+">" for phon in search_string.split(">") if "1" in phon]#[0]
    #Search string is not substring of phon_string.
    if phon_string not in search_string:
        #To avoid index out of range error.
        if stress != []:
            if stress[0] in phon_string:
                return minimum_edit_distance(search_string, phon_string)

#TODO: can pass lev-dist-function as parameter!

# 2. Levenshtein: Find phonetically close tokens. Can't be substrings.
# Different weights: stresses, close location!
# Search for min dist.
def mod_delta(p, q):
    """
    Modified delta function from nltks.metrics.aline. Included the feature
    "high" in vowels, because there was no difference between some vowels.
    @param p:
    @param q:
    @return:
    """
    #Add phonemes from IPA, that are missing in aline.
    aline.feature_matrix["ɑ"] = {'place': 'vowel', 'manner': 'vowel2',
                                 'syllabic': 'plus', 'voice': 'plus',
                                 'nasal': 'minus', 'retroflex': 'minus',
                                 'lateral': 'minus', 'high': 'low',
                                 'back': 'back', 'round': 'minus',
                                 'long': 'minus', 'aspirated': 'minus'}
    aline.feature_matrix["ʌ"] = {'place': 'vowel', 'manner': 'vowel2',
                                 'syllabic': 'plus', 'voice': 'plus',
                                 'nasal': 'minus', 'retroflex': 'minus',
                                 'lateral': 'minus', 'high': 'mid',
                                 'back': 'back', 'round': 'minus',
                                 'long': 'minus', 'aspirated': 'minus'}
    aline.feature_matrix["ɝ"] = {'place': 'vowel', 'manner': 'vowel2',
                                 'syllabic': 'plus', 'voice': 'plus',
                                 'nasal': 'minus', 'retroflex': 'plus',
                                 'lateral': 'minus', 'high': 'mid',
                                 'back': 'central', 'round': 'minus',
                                 'long': 'minus', 'aspirated': 'minus'}
    aline.feature_matrix["ɪ"] = {'place': 'vowel', 'manner': 'vowel2',
                                 'syllabic': 'plus', 'voice': 'plus',
                                 'nasal': 'minus', 'retroflex': 'minus',
                                 'lateral': 'minus', 'high': "mid",#'low-mid',
                                 'back': 'front', 'round': 'minus',
                                 'long': 'minus', 'aspirated': 'minus'}
    aline.feature_matrix["ʊ"] = {'place': 'vowel', 'manner': 'vowel2',
                                 'syllabic': 'plus', 'voice': 'plus',
                                 'nasal': 'minus', 'retroflex': 'minus',
                                 'lateral': 'minus', 'high': "mid",#'low-mid',
                                 'back': 'back', 'round': 'minus',
                                 'long': 'minus', 'aspirated': 'minus'}

    #Add feature "high" to diffentiate some vowels.
    aline.R_v = [
        "back",
        "lateral",
        "long",
        "manner",
        "nasal",
        "place",
        "retroflex",
        "round",
        "syllabic",
        "voice", "high"
    ]

    features = aline.R(p, q)
    total = 0
    for f in features:
        total += aline.diff(p, q, f) * aline.salience[f]

    return total

def arpabet_to_ipa3(phon_str: str) -> str:
    """
    Helper function that looks up an arpabet-symbol
    and returns IPA-symbol.
    @param phoneme: phoneme as arpabet-symbol.
    @return: phoneme as IPA-symbol
    """
    codes = {}
    biphones = {"AW":"aʊ", "AY":"aɪ", "CH":"tʃ", "EY":"eɪ", "JH":"dʒ",
                "OW":"oʊ", "OY":"ɔɪ"}
    phon_str = phon_str[1:-1].split("><")
    phon_str_ipa = ""

    with open('arpabet_to_ipa.csv', mode='r') as infile:
        reader = csv.reader(infile)
        for line in reader:
            line = line[0].split()
            codes[line[0]] = line[1]

    for phon in phon_str:

        #Phoneme contains stress
        if phon[-1].isnumeric():
            phon_no_stress = phon[0:-1]
            if phon_no_stress in biphones:
                for elem in biphones[phon_no_stress]:
                    phon_str_ipa += "<{}>".format(elem+phon[-1])
            else:
                phon_str_ipa += "<{}>".format(codes[phon_no_stress]+phon[-1])
        else:
            if phon in biphones:
                for elem in biphones[phon]:
                    phon_str_ipa += "<{}>".format(elem)
            else:
                phon_str_ipa += "<{}>".format(codes[phon])
    return phon_str_ipa

def sub_cost3(p:str, q:str) -> int:
    """
    Function that transforms phoneme similarity into costs for levenshtein
    function.
    @param p: phoneme as arpabet-symbol
    @param q: phoneme as arpabet-symbol
    @return: costs
    """

    p, q, sub_cost = same_stress(p, q)
    #Change weight of same stress.
    sub_cost += 2
    delta = mod_delta(p, q)
    if delta == 0:
        return sub_cost
    elif delta <= 10:
        sub_cost += 1
    elif delta > 100:
        sub_cost += 3
    else:
        sub_cost += 2
    return int(sub_cost)

def same_stress(p:str, q: str):
    """
    Helper function that takes two phonemes with stress-symbols.
    Removes stress symbols and returns cost for levenshtein function
    @param p:
    @param q:
    @return:
    """
    p_no_stress = ""
    q_no_stress = ""
    cost = 0
    if p[-1].isdigit():
        if q[-1].isdigit():
            #Phonemes with equal stresses
            if p[-1] == q[-1]:
                p_no_stress = p[0:-1]
                q_no_stress = q[0:-1]
                return p_no_stress, q_no_stress, cost
            #Phonemes with different stresses
            else:
                p_no_stress = p[0:-1]
                q_no_stress = q[0:-1]
                return p_no_stress, q_no_stress, cost+1
        #p has stress, but q does not.
        else:
            p_no_stress = p[0:-1]
            return p_no_stress, q, cost+2
    #p has no stress, but q does.
    else:
        # p has no stress, but q does.
        if q[-1].isdigit():
            q_no_stress = q[0:-1]
            return p, q_no_stress, cost+2
        #No phoneme has stress
        else:
            return p, q, cost

def iterative_levenshtein3(s, t, costs=(6, 6, 1)):
    """
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t

        costs: a tuple or a list with three integers (d, i, s)
               where d defines the costs for a deletion
                     i defines the costs for an insertion and
                     s defines the costs for a substitution
    Source: https://www.python-course.eu/levenshtein_distance.php
    """
    s = arpabet_to_ipa3(s)[1:-1].split("><")
    t = arpabet_to_ipa3(t)[1:-1].split("><")
    rows = len(s) + 1
    cols = len(t) + 1
    deletes, inserts, substitutes = costs
    dist = [[0 for x in range(cols)] for x in range(rows)]

    # source prefixes can be transformed into empty strings
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = row * deletes

    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col * inserts

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                #New implementation
                cost = sub_cost3(s[row - 1],t[col - 1])
            dist[row][col] = min(dist[row - 1][col] + deletes,
                                 dist[row][col - 1] + inserts,
                                 dist[row-1][col-1] + cost)  # substitution

    return dist[row][col]

def find_phon_levdist3(search_string,phon_string):
    """
    Find phonetically close tokens
    @param search_string:
    @param phon_string:
    @param dist:
    @return:
    """

    if phon_string not in search_string:
        return iterative_levenshtein3(search_string, phon_string)

def find_phrase_trans(sent, sourcefile, trgfile):
    sent = sent.translate(str.maketrans('', '', string.punctuation)).lower()

    results = []
    with open(sourcefile, "r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if re.search(line, sent):
                results.append(line)
    return results

def split_phonemes(phonstring:str) -> list:
    """
    Helper function that splits a phonetic token into it's phonemes
    @param phonstring:
    Example: <S><EH1><D> = "said"
    @return:
    """
    return phonstring[1:-1].split("><")

def main():

    # Idea 19.4 --> Train BPE on dictionary and generate learned mappings?

    # 1. Extract phoneme string types (homophone types).
    path_mfa_dic = "/home/user/staehli/master_thesis/homophone_analysis/mfa_output/phrases.filt.mfa"
    path_full_dic = "/home/user/staehli/master_thesis/homophone_analysis/mfa_output/english.dict"
    phrases_vocab = "/home/user/staehli/master_thesis/homophone_analysis/mfa_input/phrases.filt.vocab.en"
    train_mfa_dic = pde.get_dictionary(path_mfa_dic)
    full_dic = pde.get_dictionary(path_full_dic)
    #Phonetized vocabulary
    vocab_phon = pde.vocab_to_dictionary(phrases_vocab, full_dic, train_mfa_dic)

    #Homophones in train data. TODO: Write this to file, because programm runs slow.
    homophones_en = pde.get_homophone_tuples(vocab_phon)
    print("Length of phonetized vocabulary:", len(vocab_phon))
    print("Length of homophone types in data:", len(homophones_en))
    counter = 0
    for key in homophones_en:
            if len(homophones_en[key]) > 1:
                print(homophones_en[key])
                counter += 1
    #             counter += len(homophones_en[key])
    print("Number of homophones with 2 or more grapheme types:", counter)
    # print(len(homophones_en["<UNK>"]))

    # 1.b Dictionary as dataframe
    # train_dataframe_en = pd.DataFrame(homophones_en.items(), columns=["phoneme type", "grapheme type"])
    # train_dataframe_en.to_csv("train.en.freq.csv")

    # 2. Generate phoneme representations by sentence and write to file (source side).
    # src_phrases = "/home/user/staehli/master_thesis/homophone_analysis/phrases.filt.en"
    # trg_phrases = "/home/user/staehli/master_thesis/homophone_analysis/phrases.filt.de"
    # pde.grapheme_to_phoneme(vocab_phon, src_phrases, trg_phrases, "phrases.filt.ph.en")

    # TODO: Fix compunds with hypen (split before training MFA).
    # counter3 = 0
    # for value in homophones_en["<UNK>"]:
    #     if "-" in value:
    #         value = value.split("-")
    #         for token in value:
    #             if token not in vocab_phon:
    #                 print(token)


    ### Edit distance with "normal" levenshtein. ###

    # profiler = cProfile.Profile()
    # profiler.enable()
    #
    # homophones = [key for key in homophones_en if key != "<UNK>"][0:1000]
    # similar = ((key, match) for key, match in itertools.combinations(homophones, 2) if find_phon_levdist(key, match) == 1)
    # for tup in similar:
    #     print(tup)
    #
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("tottime")
    # stats.print_stats()

    ### Edit distance with "normal" levenshtein. ###

    ### Edit distance with "weighted" levenshtein ###

    # profiler = cProfile.Profile()
    # profiler.enable()
    # counter2 = 0
    # for key in homophones_en:
    #     if key != "<UNK>":
    #         if find_phon_levdist3(ex, key, 2*6):
    #             print(key, homophones_en[key])
    #             counter2 +=1
    # print("\n")
    # print("Similar strings f1:", counter1, "Similar strings f2:", counter2)

    # homophones = [key for key in homophones_en if key != "<UNK>"][
    #              0:1000]
    # similar = ((key, match) for key, match in
    #            itertools.combinations(homophones, 2) if
    #            find_phon_levdist3(key, match) == 1)
    # for tup in similar:
    #     print(tup)
    #
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("tottime")
    # stats.print_stats()

    ### Edit distance with "weighted" levenshtein ###


    ### Find phonetically similar phrases ###
    # phrases_en = "/home/user/staehli/master_thesis/homophone_analysis/phrases.en"
    # phrases_de = "/home/user/staehli/master_thesis/homophone_analysis/phrases.de"
    # source = "In other words, play is our adaptive wildcard."
    # reference = "Mit anderen Worten, Spiel ist unser Anpassungsjoker."
    # target = "Anders gesagt , Orte sind anpassungsfähig ."
    # for phrase in find_phrase_trans(source,phrases_en, phrases_de):
    #     print(phrase)
    # print(linecache.getline("phrases.ph.en", 3806))


    ### Testint finding similar phonetic phrases ###
    # sent ="<PHON:<<IH1><N>>> <PHON:<<AH1><DH><ER0>>> <PHON:<<W><ER1><D><Z>>>"
    # sent = sent[0:-1].replace("<PHON:", "").replace("> ", "")
    # sent2 = "<PHON:<<W><IY1>>> <PHON:<<S><EH1><D>>>" # "we said"
    #
    # f = open("phrases.ph.en", "r")
    # for line in its.iter_sentences(f):
    #     line = line[0:-1].replace("<PHON:", "").replace("> ", "")
    #     dist = minimum_edit_distance(sent, line)
    #     if dist <= 2:
    #         print(line)
    # f.close()

    # print(sent)

    #TODO: print out grapheme representations of everything --> lookup with original file.

    #Compare this result against working only with grapheme strings!

    # I should focus on homophone string on the source side. Matching to translations on the target side is hard!

if __name__ == "__main__":
    main()