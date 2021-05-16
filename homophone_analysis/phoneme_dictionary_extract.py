#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

import argparse
import typing
from typing import Iterator
from itertools import chain
from collections import Counter
import string

class PhonDicExtract:
    """
    class to extract phonetic representations of types from a MFA output file.
    """
    def __init__(self, filename=""):
        self.__filename__ = filename
        self.__mappings__ = dict()
        self.__set_mappings__()
        self.get_dictionary()

    def __data_formatter__(self) -> Iterator:
        """
        Method that formats the lines of MFA pronunciation dictionary to be
        further processed.
        @return:
        """
        with open(self.__filename__, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.rstrip().split(maxsplit = 1)
                yield line

    def __set_mappings__(self):
        #TO DO: list as value not needed, because MFA outputs 1 possible representation.
        #Change values to type str in all methods!
        for elem in self.__data_formatter__():
            self.__mappings__[elem[0]] = elem[1]

    def get_dictionary(self):
        return self.__mappings__

def main():
    source_mfa = "/home/user/staehli/master_thesis/homophone_analysis/mfa_output/source_lower_dictionary"
    source = PhonDicExtract(filename=source_mfa)
    source_dic = source.get_dictionary()
    source_homophones = get_multiple_pairings(source_dic)
    print(len(get_multiple_pairings(source_dic)))
    for key in source_homophones:
        print(key, source_homophones[key])

    # source_test = "/home/user/staehli/master_thesis/homophone_analysis/mfa_input/test.src.lower.en"
    # target_test = "/home/user/staehli/master_thesis/homophone_analysis/mfa_input/result.lower.de"
    #
    # homophones = [hphone for hphones in source_homophones.values() for hphone in hphones]
    # print(homophones)
    #
    # for homophone in homophones:
    #     print_homophone_translations(homophone, source_test, target_test)
    # print_homophone_translations("meat", source_test, target_test)
    # Print sentences that contain homophones and their translation.


    ### OBSERVATIONS ###
    #Characters not recognized by MFA: ! # % & ( ) , . 0 1 2 3 4 5 6 7 8 9 : ; ? @ á ã ♫

    # I could install Levenshtein library to measure distance between
    #types that are not shared between the two dictionaries!

    # Levenshtein distance (match word, check phonemes) to see if difference in vocabulary
    # is due to misheard word or simply different translation?

    # Overlaps in phoneme representation that doesn't show up in grapheme representation would
    # be interesting. Are there any cases?

    #How can I check if the system is better or worse (because of attention)
    # at the beginning or end of a word?

if __name__ == "__main__":
    main()