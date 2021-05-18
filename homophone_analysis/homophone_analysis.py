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


def get_multiple_pairings(dic: dict) -> dict:
    """
    Method that shows multiple mappings between grapheme and phoneme
    strings.
    @dic: dict {type: [phoneme string type1, phoneme string type2]}
    @return: dictionary {phoneme string type: [type1, type2]}
    """
    #Solution by:
    # https://www.geeksforgeeks.org/python-find-keys-with-duplicate-values-in-dictionary/
    rev_dict = {}
    multiples = dict()

    for key, value in dic.items():
        rev_dict.setdefault(value, set()).add(key)

    result = set(chain.from_iterable(
        values for key, values in rev_dict.items()
        if len(values) > 1))
    for r in result:
        for key, value in dic.items():
            if r == key:
                if value not in multiples:
                    multiples[value] = [key]
                else:
                    multiples[value].append(key)

    return multiples

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

def get_alignment(sent_id: int, word_id: int, alignment_file: str) -> list:
    """
    Helper function that retrieves the alignment of source_token with
    target_token in sent_n
    @param sent_id: sentence number in source file.
    @param word_id: position of token in source sentence.
    @param alignment_file: fast_align alignment file.
    @return:
    """
    with open(alignment_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
        for i in range(len(lines)):
            if i == sent_id:
                line = [elem.split("-") for elem in lines[i].split()]
                new_line = [[int(s) for s in sublist] for sublist in line]
                for j in new_line:
                    if j[0] == word_id:
                        return j

def get_homophone_translations(token, source, translation, alignment_file):
    """
    Function that opens a source and reference/target file of aligned
    sentences and prints sentences that contains token on source side.
    @param token:
    @param source:
    @param translation:
    @return:
    """
    #STEPS:
    #1. Iterate over files in parallel
    #2. Find homophone string in source.
    #3. Helper function that retrieves alignment for sent_id/word_id --> string to list --> list to dictionary
    #4. Print searched homophone words and translation of word + sliding window.
    #TODO: sliding window + Write to file --> I want to search/view homophone representations!
    with open(source, "r", encoding="u8") as src, \
         open(translation, "r", encoding="u8") as trans:
        src = src.readlines()
        trans = trans.readlines()
        for i in range(len(src)):
            src_line = src[i]
            if token in src_line:
                for (j, k) in enumerate(src_line.split()):
                    if token == k:
                            alignment = get_alignment(i, j, alignment_file)
                            print(alignment)
                            try:
                                trans_line = trans[i].split()
                                translation = trans_line[alignment[1]]
                                print("THE HOMOPHONE IS:", token)
                                print("THE TRANSLATION IS:", translation)
                                print("\n")
                            except TypeError:
                                #PRINT source and translation.
                                #Fast align didn't align correctly.
                                #Could also be due to poor ASR: I don't want to discard these cases! These interest me!
                                print("ALIGNMENT NOT FOUND")
                                print(src_line)
                                print(trans_line)
                                print("\n")
                                break


def main():
    # I want this to compare over word boundaries, but I don't know if this will work.
    # Try to match ngrams and use brevity penalty for longer sentences?
    # Highlight overlaps in phones? How do I filter out garbage?
    # How do I ensure high precision (and recall)?
    # Idea 19.4 --> Train BPE on dictionary and generate learned mappings?

    # 1. Extract phoneme string types (homophone types).
    source_mfa = "/home/user/staehli/master_thesis/homophone_analysis/mfa_output/source_lower_dictionary"
    source = PhonDicExtract(filename=source_mfa)
    source_dic = source.get_dictionary()
    source_homophones = get_multiple_pairings(source_dic)
    # print(len(get_multiple_pairings(source_dic)))
    # for key in source_homophones:
    #     print(key, source_homophones[key])

    # 2. Generate phoneme representations by sentence.
    test_lc_en = "/home/user/staehli/master_thesis/homophone_analysis/mfa_input/test.lc.en"
    # grapheme_to_phoneme(source_dic, test_lc_en, "test.ph.en", concat=True)

    # 3. Search two homophone types in source/reference (sentence match).
    test_tc_en = "/home/user/staehli/master_thesis/data/MuST-C/test.tc.en"
    test_tc_ref_de = "/home/user/staehli/master_thesis/data/MuST-C/test.tc.de"
    # get_homophone_translations("knew", test_tc_en, test_tc_ref_de)
    # get_homophone_translations("new", test_lc_en, test_lc_ref_de)

    # 4. Search two homophone types in source/reference (word alignment).
    alignment_file = "/home/user/staehli/master_thesis/homophone_analysis/alignments/forward.lc.src-ref.align"
    # get_alignment(0, 1, alignment_file)
    get_homophone_translations("knew", test_tc_en, test_tc_ref_de, alignment_file)


    # TODO: import extract homophones and check how many times each
    # homophone type appears in test data.

if __name__ == "__main__":
    main()