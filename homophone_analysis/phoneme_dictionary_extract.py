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
    Class to extract phonetic representations of types from a MFA output file.
    """
    def __init__(self, filename=""):
        self.__filename__ = filename

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

    def get_dictionary(self) -> dict:
        """
        Method that returns a dictionary with {grapheme string type: [phoneme string type1, phoneme string type2]}
        @return:
        """
        # TODO: list as value not needed, because MFA outputs 1 possible representation.
        # Change values to type str in all methods!
        mappings = {}
        for elem in self.__data_formatter__():
            mappings[elem[0]] = elem[1]
        return mappings

    def get_homophone_tuples(self, dic: dict) -> dict:
        """
        Method that shows multiple mappings between grapheme and phoneme
        strings.
        @dic: dict {type: [phoneme string type1, phoneme string type2]}
        @return: dictionary {phoneme string type: [type1, type2]}
        """
        # Solution by:
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

def main():
    source_mfa = "/home/user/staehli/master_thesis/homophone_analysis/mfa_output/source_lower_dictionary"
    source = PhonDicExtract(filename=source_mfa)
    source_dic = source.get_dictionary()
    source_homophones = source.get_homophone_tuples(source_dic)
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