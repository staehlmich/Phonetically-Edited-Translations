#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

import re
import typing
import string
from collections import OrderedDict
import operator

"""Script to filter and format output of moses phrase table.
Next step: Run G2P (MFA) on filtered phrase table."""

def line_iter(filename: str):
    """
    Helper function to read lines from moses phrase table.
    @param filename:
    @return:
    """
    with open(filename, "r", encoding="utf-8") as infile:
        for line in infile:
            line = [elem.strip() for elem in line.rstrip().split("|||")]
            line = [elem for elem in line if elem != ""]
            #Exclude phrases that contain punctuation, sp.
            if isalphaspace(line[1]) == True:
                if isalphaspace(line[0]) == True:
                    srce = line[1]
                    trgt = line[0]
                    prob = line[2]
                    freq = line[4]
                    yield srce, trgt, prob, freq

def isalphaspace(phrase: str) -> bool:
    """
    Helper function that determines if a string is all alphabetical
    characters, whitespaces and apostrophes.
    @param phrase: phrase from moses phrase table (en or de).
    @return:
    """
    if all(x.isalpha() or x.isspace() or x == "'" for x in phrase):
        #Exclude phrases that are split at apostrophe.
        if phrase.startswith("'") == False:
            return True
    else:
        return False

def filter_table(table_path:str, min_freq=1, n=5, tscore=0.05) -> dict:
    """
    Function that filters moses phrase table and writes aligned phrases
    to separate files.
    @param table_path: Path to moses phrase table.
    @param min_freq: mininum frequency of phrase.
    @param n: number of translations per source phrase.
    @param tscore: threshold of translation probabiliy.
    @return: ordered dictionary with top n translations per phrase.
    """
    phrases = OrderedDict()
    for srce, trgt, prob, freq in line_iter(table_path):
        prob = prob.split()
        freq = freq.split()
        #Filter only phrases with 3 or less tokens.
        if srce.count(" ") < 3:
            #Filter by frequency of english phrase.
            if float(freq[0]) > min_freq:
                #Check if translation not english phrase.
                if srce != trgt:
                    #Format element for dictionary.
                    new_phrase = (float(prob[0]), float(prob[2]), trgt)
                    # Filter phrases by inverse and direct
                    # translation probability.
                    if all(elem > tscore for elem in new_phrase[0:2]):
                        if srce not in phrases:
                            phrases[srce] = [new_phrase]
                        else:
                            phrases[srce].append(new_phrase)

    #Keep only top n translations.
    for key, value in phrases.items():
        sorted_trans = sorted(value, key=operator.itemgetter(0,1), reverse=True)
        phrases[key] = sorted_trans[:n]

    return phrases

def write_filtered_table(filtered_dic: dict, filename_filtered:str, joined=True):
    """
    Write filtered table to file.
    @param filtered_dic: filtered phrase table as dictionary.
    @param filename_filtered:str: Write filtered phrase table to this path.
    If joined file contains: "source_phrase ||| target_phrase\n"
    Else file contains: "source/target phrase\n"
    @param joined: If true, writes source and target phrases to same file.
    @return:
    """
    if joined:
        with open(filename_filtered+".en-de", "w", encoding="utf-8") as out:
            for key in filtered_dic:
                #Write all translations to the same line.
                out.write(key + " ||| " + "/".join(trans[2] for trans in filtered_dic[key]) + "\n")
    else:
        with open(filename_filtered+".en", "w", encoding="utf-8") as src, \
          open(filename_filtered + ".de", "w", encoding="utf-8") as trg:
            for key in filtered_dic:
                src.write(key + "\n")
                #Write all translations to the same line.
                trg.write("/".join(trans[2] for trans in filtered_dic[key]) + "\n")


def main():
    #TODO: argparse.
    phrase_table_path = "/home/user/staehli/master_thesis/homophone_analysis/moses_experiments/model/phrase-table.detok"
    # phrase_table_short = "/home/user/staehli/master_thesis/homophone_analysis/moses_experiments/model/phrases.detok.short"

    filtered_table = filter_table(phrase_table_path)

    # write_filtered_table(filtered_table, "phrases.filtered.en", "phrases.filtered.de")
    write_filtered_table(filtered_table, "phrases.filtered4", joined=True)

if __name__ == "__main__":
    main()