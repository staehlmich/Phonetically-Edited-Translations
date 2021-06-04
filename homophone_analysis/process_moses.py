#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli
import re
import typing
import string
"""Script to format output of moses phrase table to train with MFA."""

def find_substrings(s: str, l: list):
    """
    Helper function to find substrings of strings in list
    @param s: search string
    @param l: list containing string
    @return: bool
    """
    substring = False
    if s in l:
        substring = True
        return substring
    for elem in l:
        if s in elem:
            substring = True
            return substring
        else:
            pass
    return substring

def main():
    phrase_table_path = "/home/user/staehli/master_thesis/homophone_analysis/moses_experiments/model/phrase-table"

    #Open moses output file from step 6.
    #Number of translation per source phrase
    n = 3
    with open(phrase_table_path, "r", encoding="utf-8") as infile:
        phrases = {}
        # counter = 0
        previous_phrase = ""
        for line in infile:
            line = line.rstrip().split("|||")
            #Source phrase is first element.
            source, trans, score = line[1].strip(), line[0].strip(), line[2].split()[2]
            # counter += 1
            # if counter < 20:
            #Check if source phrase is not a punctuation symbol.
            if re.search('[a-zA-Z]', source):
                # Filter by score.
                if float(score) > 0.5:
                    if source not in phrases:
                        phrases[source] = {float(score): trans}
                        #Check if phrases already contains a phrase to
                        #avoid KeyError.
                        if len(phrases) > 1:
                            # Current line is substring of previous line.
                            if source in previous_phrase:
                                del phrases[source]
                            else:
                                previous_phrase = source
                        else:
                            previous_phrase = source

                    else:
                        if len(phrases[source]) < n:
                            phrases[source].update({float(score): trans})
                        else:
                            lowest_score = min(phrases[source].keys())
                            del phrases[source][lowest_score]
                            phrases[source].update({float(score): trans})

        #Delete duplicate phrase pairs.
        # phrases = set(phrases)
        #Write phrases to files.
        with open("phrases.en", "w", encoding="utf-8") as phrases_source, open("phrases.de", "w", encoding="utf-8") as phrases_target:
            #Keep both phrases for alignment
            for key in phrases:
                for value in phrases[key].keys():
                    #Write source phrase multiple times, to align with
                    #multiple translation posibilities.
                    phrases_source.write(key+ "\n")
                    phrases_target.write(phrases[key][value]+"\n")

            #Filter out substrings.
            #Solution by: https://stackoverflow.com/questions/22221878/delete-substrings-from-a-list-of-strings

if __name__ == "__main__":
    main()