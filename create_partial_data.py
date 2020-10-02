#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Michael Staehli

import argparse
from typing import Iterator
from itertools import islice
import re

class PartialDataGenerator(object):
    """
    Class that creates partial sentences from training data.
    """
    def __init__(self, filename):
        self.__filename__= filename
        # self.__format_data__()

    def __data_reader__(self)-> Iterator:
        """
        Method that reads a training file and yields lines of data.
        @return:
        """
        with open(self.__filename__, "r", encoding="utf-8") as infile:
            for line in infile:
                yield line

    def __generate_triplets__(self)-> Iterator:
        """
        Method that yields 3 lines at a time from training data.
        The 3 lines are a GIZA++ relevant format.
        @return:
        """
        #TO DO: How to use islice and yield multiple triplets?
        # with open(self.__filename__, "r", encoding="utf-8") as infile:
        triplet_generator = islice((line for line in self.__data_reader__()), 3)
        return triplet_generator
        # for triplet in triplet_generator:
        #     print(triplet)
            # for triplet in triplet_generator:
            #     yield triplet

    def __format_data__(self) -> Iterator:
        """
        Method that yields alignments as list (target) and dictionary
        source).
        @return: target, source.
        """
        target = []
        source = {}
        for i, elem in enumerate(self.__generate_triplets__()):
            if i == 1:
                target = elem.lstrip().split(" ")
                #Dirty fix: delete newline char in sentence list.
                if target[-1] == "\n":
                    target.pop()
            if i == 2:
                source = self.__create_align_dic__(elem)
            #Two easy conditions, so that we don't yield empty elems.
            if target != []:
                if source != {}:
                    yield target, source

    def __create_align_dic__(self, source_sent: str) -> dict:
        """
        Method that creates a didctionary with source words (values)
        and alignemnt positions (keys).
        @param source_sent: source sentence in GIZA++ format as string.
        @return:
        """
        source = source_sent.rstrip("").split(")")
        alignment_dic = {}
        for elem in source:
            #Counter for unaligned source (?) words.
            counter = 900
            elem = elem.lstrip().rstrip()
            if elem != "":
                source_word = elem.split("(")[0].rstrip(" ")
                #Idea found in this post:
                #https://stackoverflow.com/questions/4289331/how-to-extract-numbers-from-a-string-in-python
                positions=[int(s) for s in re.findall(r'\b\d+\b', elem.split("(")[1])]
                #Since there can be multiple alignments, we iterate over
                # list with positions.
                #It is possible to have NULL alignments in both directions!
                #Words don't work as values, because tokens can be repeated in
                #a sentence.
                if positions == []:
                    counter += 1
                    alignment_dic[counter] = source_word
                else:
                    for pos in positions:
                        alignment_dic[pos] = source_word

        return alignment_dic

    def generate_partial_sents(self):
        """

        @return:
        """
        for target, source in self.__format_data__():
            #Target and source contain "\n" with (EOS).
            aligned_target = ""
            aligned_source = ""
            for i in range(len(target)):
                # Exclude last element, because we don't have to add "\w" at EOS.
                if i != len(target):
                    #Index for source sentence.
                    j = i+1
                    try:
                        aligned_target += target[i] + " "
                        aligned_source += source[j] + " "
                    except KeyError:
                        print(source, target)
                        # aligned_target += target[i]
                        # aligned_source += source[j]
                    # print(aligned_target)
                    # print(aligned_source)
                    # print("\n")
                else:
                    aligned_target += target[i]
                    aligned_source += source[j]
                    #This is the final yield of full sentences.
                    print(aligned_target)
                    print(aligned_source)
                    print("\n")

#GIZA++ output:
#Line1: target length  and source length.
#Line2: target sentence.
#Line3: source sentence with with alingment information
#NULL --> Single brackets, e.g. NULL {6} = 6th target word not aligned.
#Double brackets, e.g. Vi {1 2} = 1st and 2nd target word aligned to "Vi".


def main():
    parser = argparse.ArgumentParser(
        description="Generate partial sentence training data.")
    parser.add_argument("-f", "--filename", type=str,
                        help="Path to file.")
    args = parser.parse_args()
    PDG = PartialDataGenerator(args.filename)
    # PDG.generate_partial_sents()
    for line in PDG.__data_reader__():
        print(line)
#1. Read and understand algorithm Niehues et al. (2018)
#2. Undertand how to read and use GIZA++ alignments.
#3. Create partial data with ted data and alignment info.

if __name__ == "__main__":
    main()