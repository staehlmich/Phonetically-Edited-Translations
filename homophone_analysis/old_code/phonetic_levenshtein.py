#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

from nltk.metrics import aline
from nltk.metrics import edit_distance
import cProfile
import pstats
import typing
import timeit
import Levenshtein
import numpy as np
from collections import Counter
import csv
import pprint

class SubCost(object):
    """
    Class to calculate the substitution cost of two phonemes for Levenshtein
    function. Costs are due to phonetic similarity, calculated with
    nltk.metrics.aline.
    """
    def __init__(self, phon1: str, phon2:str):
        #Add missing features to aline.
        self._aline_add_features()
        #Map arpabet to IPA.
        self._mappings = self._set_mappings()
        self._phon1 = phon1
        self.phon2 = phon2
        self._sub_cost = 0
        self._aline_add_features()
    def _aline_add_features(self):
        """
        Method that adds features for missing arpabet phonemes.
        @return:
        """
        # Add phonemes from IPA, that are missing in aline.
        aline.feature_matrix["ɑ"] = {'place': 'vowel',
                                     'manner': 'vowel2',
                                     'syllabic': 'plus',
                                     'voice': 'plus',
                                     'nasal': 'minus',
                                     'retroflex': 'minus',
                                     'lateral': 'minus', 'high': 'low',
                                     'back': 'back', 'round': 'minus',
                                     'long': 'minus',
                                     'aspirated': 'minus'}
        aline.feature_matrix["ʌ"] = {'place': 'vowel',
                                     'manner': 'vowel2',
                                     'syllabic': 'plus',
                                     'voice': 'plus',
                                     'nasal': 'minus',
                                     'retroflex': 'minus',
                                     'lateral': 'minus', 'high': 'mid',
                                     'back': 'back', 'round': 'minus',
                                     'long': 'minus',
                                     'aspirated': 'minus'}
        aline.feature_matrix["ɝ"] = {'place': 'vowel',
                                     'manner': 'vowel2',
                                     'syllabic': 'plus',
                                     'voice': 'plus',
                                     'nasal': 'minus',
                                     'retroflex': 'plus',
                                     'lateral': 'minus', 'high': 'mid',
                                     'back': 'central',
                                     'round': 'minus',
                                     'long': 'minus',
                                     'aspirated': 'minus'}
        aline.feature_matrix["ɪ"] = {'place': 'vowel',
                                     'manner': 'vowel2',
                                     'syllabic': 'plus',
                                     'voice': 'plus',
                                     'nasal': 'minus',
                                     'retroflex': 'minus',
                                     'lateral': 'minus', 'high': "mid",
                                     # 'low-mid',
                                     'back': 'front', 'round': 'minus',
                                     'long': 'minus',
                                     'aspirated': 'minus'}
        aline.feature_matrix["ʊ"] = {'place': 'vowel',
                                     'manner': 'vowel2',
                                     'syllabic': 'plus',
                                     'voice': 'plus',
                                     'nasal': 'minus',
                                     'retroflex': 'minus',
                                     'lateral': 'minus', 'high': "mid",
                                     # 'low-mid',
                                     'back': 'back', 'round': 'minus',
                                     'long': 'minus',
                                     'aspirated': 'minus'}

        # Add feature "high" to diffentiate vowels.
        aline.R_v.append("high")

    def _set_mappings(self):
        """
        Return a dictionary mapping arpabet codes to IPA codes.
        Source: https://en.wikipedia.org/wiki/ARPABET
        @return:
        """

        d = {'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ',
             'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
             'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'g',
             'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
             'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
             'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
             'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
             'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'}
        return d

    def _same_stress(self, p: str, q: str):
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
                # Phonemes with equal stresses
                if p[-1] == q[-1]:
                    p_no_stress = p[0:-1]
                    q_no_stress = q[0:-1]
                    return p_no_stress, q_no_stress, cost
                # Phonemes with different stresses
                else:
                    p_no_stress = p[0:-1]
                    q_no_stress = q[0:-1]
                    return p_no_stress, q_no_stress, cost + 1
            # p has stress, but q does not.
            else:
                p_no_stress = p[0:-1]
                return p_no_stress, q, cost + 2
        # p has no stress, but q does.
        else:
            # p has no stress, but q does.
            if q[-1].isdigit():
                q_no_stress = q[0:-1]
                return p, q_no_stress, cost + 2
            # No phoneme has stress
            else:
                return p, q, cost

    def _arpabet_to_ipa3(self, phon_str: str) -> str:
        """
        Helper function that looks up an arpabet-symbol
        and returns IPA-symbol.
        @param phoneme: phoneme as arpabet-symbol.
        @return: phoneme as IPA-symbol
        """
        codes = {}
        biphones = {"AW": "aʊ", "AY": "aɪ", "CH": "tʃ", "EY": "eɪ",
                    "JH": "dʒ",
                    "OW": "oʊ", "OY": "ɔɪ"}
        phon_str = phon_str[1:-1].split("><")
        phon_str_ipa = ""

        for phon in phon_str:

            # Phoneme contains stress
            if phon[-1].isnumeric():
                phon_no_stress = phon[0:-1]
                if phon_no_stress in biphones:
                    for elem in biphones[phon_no_stress]:
                        phon_str_ipa += "<{}>".format(elem + phon[-1])
                else:
                    phon_str_ipa += "<{}>".format(
                        codes[phon_no_stress] + phon[-1])
            else:
                if phon in biphones:
                    for elem in biphones[phon]:
                        phon_str_ipa += "<{}>".format(elem)
                else:
                    phon_str_ipa += "<{}>".format(codes[phon])
        return phon_str_ipa

    def _mod_delta(self):
        """
        Modified delta function from nltk.metrics.aline. Included the feature
        "high" in vowels, because there was no difference between some vowels.
        @return: phone similarity measure.
        """
        features = aline.R(self._phon1, self._phon2)
        total = 0
        for f in features:
            total += aline.diff(self._phon1, self._phon1, f) * aline.salience[f]

        return total

    def get_sub_cost(self, p: str, q: str) -> int:
        """
        Function that transforms phoneme similarity into costs for levenshtein
        function.
        @param p: phoneme as arpabet-symbol
        @param q: phoneme as arpabet-symbol
        @return: costs
        """

        p, q, sub_cost = same_stress(p, q)
        # Change weight of same stress.
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

def minimum_edit_distance(source, target):
    """
    Returns the minimum edit distance between two strings @param source and
    @param target, given the cost of
        insertion    = 1;
        deletion     = 1;
        substitution = 1.
    """
    n = len(source)
    m = len(target)
    d = [[None for _ in range(m+1)] for _ in range(n+1)]
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

def iterative_levenshtein(s, t, costs=(1, 1, 1)):
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
    """

    rows, cols = len(s) + 1, len(t) + 1
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
            dist[row][col] = min(dist[row - 1][col] + deletes,
                                 dist[row][col - 1] + inserts,
                                 dist[row - 1][
                                     col - 1] + (substitutes if s[row-1] != t[col-1] else 0))  # substitution

    return dist[rows-1][cols-1]

def fast_levdist(search, target, costs=(1,1,1)):
    ops = Counter(elem[0] for elem in Levenshtein.editops(search, target))
    dels, ins, subs = costs
    #Still need to know which characters get replaced, for phonetic similarity.
    dist = ops["replace"]*subs+ops["insert"]*ins+ops["delete"]*dels

    return dist

def main():
    search = "<P><L><EY1><S>"
    target = "<F><L><EY1><M>"
    print(timeit.timeit('minimum_edit_distance("search", "target")', number=100000, globals=globals()))
    print(timeit.timeit('iterative_levenshtein("search", "target")', number=100000, globals=globals()))
    print(timeit.timeit('Levenshtein.distance("search", "target")', number=100000, globals=globals()))
    print(timeit.timeit('fast_levdist("search", "target")', number=100000, globals=globals()))

    #
    # print("min edit dist:", minimum_edit_distance(search, target))
    # print("Edited levdist:", iterative_levenshtein(search, target))

    # sub = SubCost(search, target)
    # print(sub, len(sub))

if __name__ == "__main__":
    main()