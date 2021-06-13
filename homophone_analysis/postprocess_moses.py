#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli
import re
import typing
import string

"""Script to format output of moses phrase table to train with MFA."""

def remove_punct(text: str) -> str:
    """
    Helper function to remove all punctuation from string.
    Solution by:
    https://stackoverflow.com/questions/18429143/strip-punctuation-with-regex-python
    @param text: phrase as output from moses phrase table.
    @return:
    """
    return ' '.join(word.strip(string.punctuation) for word in text.split()).strip()

def replace_special_chars(special_char: str):
    """
    Helper function to replace special characters from moses tokenizer.
    @param special_char:
    @return:
    """
    #TODO as list, change search direction and keep mosees tag.
    special_chars = {"&amp;": "&", "&#124;": "|", "&lt;": "<", "&gt;": ">",
                     "&apos;": "'", "&quot;": '"', "&#91;": "[","&#93;": "]" }

    hit = None

    for key in special_chars.keys():
        if re.match(key, special_char):
            hit = True
            return hit
        else:
            hit = False
    return hit

def line_iter(filename: str):
    """
    Helper function to read lines from moses phrase table.
    @param filename:
    @return:
    """
    with open(filename, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.rstrip().split("|||")
            # Source phrase is first element.
            source, trans, score = line[1].strip(), line[0].strip(), \
                                   line[2].split()[2]
            yield source, trans, score

def filter_table(table_path:str, filename_source:str, filename_target:str, n:int, tscore = 0.5):
    """
    Function that filters moses phrase table and writes aligned phrases
    to separate files.
    @param table_path:
    @param filename_source:
    @param filename_target:
    @param n: number of translations per source phrase.
    @return:
    """

    phrases = {}
    with open(filename_source, "w", encoding="utf-8") as phrases_source, \
       open(filename_target, "w", encoding="utf-8") as phrases_target:
        for source, trans, score in line_iter(table_path):
            if float(score) > tscore:
                #Remove punctuation from source and translation.
                # We don't need it for analysis?
                source_no_punct = remove_punct(source)
                trans_no_punct = remove_punct(trans)
                if source_no_punct not in phrases:
                    phrases[source_no_punct] = {float(score):trans_no_punct}
                else:
                    #Check if translation not already in subdict.
                    if trans_no_punct not in phrases[source_no_punct].values():
                        if len(phrases[source_no_punct]) < n:
                            phrases[source_no_punct].update({float(score):trans_no_punct})
                        else:
                            #Remove translation pair with lowest score.
                            del phrases[source_no_punct][min(phrases[source_no_punct].keys())]
                            phrases[source_no_punct] = {float(score): trans_no_punct}

        # Keep both phrases for alignment
        previous_phrase = ""
        for key in phrases:
            for value in phrases[key].keys():
                if key != previous_phrase:
                    # Check if current phrase not a substring of previous phrase.
                    if key not in previous_phrase:
                        # Write source phrase multiple times, to align with
                        # multiple translation posibilities.
                        phrases_source.write(key + "\n")
                        phrases_target.write(phrases[key][value] + "\n")
                        previous_phrase = key

def main():
    phrase_table_path = "/home/user/staehli/master_thesis/homophone_analysis/moses_experiments/model/phrase-table.detok"

    #Open moses phrase table from step 6.
    filter_table(phrase_table_path, "phrases2.en", "phrases2.de", 3)

if __name__ == "__main__":
    main()