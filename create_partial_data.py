#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Michael Staehli

import argparse
from typing import Iterator
from itertools import islice



class PartialDataGenerator(object):
    """
    Class that creates partial sentences from training data.
    """
    def __init__(self, filename):
        self.__filename__= filename
        self.__format_data__()

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
        with open(self.__filename__, "r", encoding="utf-8") as infile:
            triplet_generator = islice(infile, 3)
            for triplet in triplet_generator:
                yield triplet

    def __format_data__(self):
        """

        @return:
        """
        triplets = self.__generate_triplets__()
        for triplet in triplets:
            print(triplet)

    # metainfo = None
    # target = []
    # source = []

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

#1. Read and understand algorithm Niehues et al. (2018)
#2. Undertand how to read and use GIZA++ alignments.
#3. Create partial data with ted data and alignment info.

if __name__ == "__main__":
    main()