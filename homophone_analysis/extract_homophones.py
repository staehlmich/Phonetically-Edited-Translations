#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

import argparse
import re
from collections import Counter
import typing
import pandas as pd

class HomExt:
    """
    class to extract homophones from training and test files.
    """

    def __init__(self, homophone_file:str, filename_data:str, extract = False):
        self.__homophone_file__ = homophone_file
        self.__filename_data__ = filename_data
        self.__filename_counts__ = "homophone_counts.tsv"
        self.__homophone_tuples__ = []
        self.__get_homophone_tuples__()
        # Create file with counts, if not available.
        if extract == True:
            self.__write_homophones_stats__()
        self.__homophone_stats__ = None
        # Create dataframe from counts, if not available.
        if self.__homophone_stats__ == None:
            self.__homophone_stats__ = pd.read_csv(self.__filename_counts__,
        sep="\t",
        names = ["Homophone", "Homophone_Tuple", "Sentence_ID", "Word_ID"])

    def __get_homophone_tuples__(self):
        """
        Read homophone file and save homophone tuples to list.
        @param file: File that contains list of homonyms.
        @return:
        """
        with open(self.__homophone_file__, "r", encoding="utf-8") as infile:
            for line in infile:
                hphones = line.rstrip().split(" / ")
                self.__homophone_tuples__.append(hphones)

    def __write_homophones_stats__(self):
        """
        Method that searches for homophones in test or training data and
        writes them into a .tsv file.
        Format of file: hphone, hphone tuple, sent_id, word_id
        @return:
        """
        with open(self.__filename_data__, "w", encoding="utf-8") as outfile:
            for tup in self.__homophone_tuples__:
                for hphone in tup:
                    with open(self.__filename_data__, "r",
                              encoding="utf-8") as infile:
                        # Initialize counter for sentence id.
                        counter = 0
                        for line in infile:
                            counter += 1
                            line = line.rstrip().split()
                            #Use range function to get word_id.
                            for i in range(len(line)):
                                if line[i] == hphone:
                                    t= " ".join(e for e in tup)
                                    #list with: (hphone, hphone tuple, sent_id, word_id)
                                    #Convert integers to strings.
                                    outfile.write(hphone+"\t"+t+"\t"+str(counter)+"\t"+str(i+1)+"\n")

    def get_stats(self):
        """

        @return: counts of homophones and pairs in data as pd.dataframe
        """
        return self.__homophone_stats__

#6. sentence lookup and word highlight?

def main():
    hphones = HomExt("english-homophones.txt", "/home/user/staehli/master_thesis/data/MuST-C/test.tc.en")
    stats = hphones.get_stats()

    # Questions for data:
    # 1. Which are the 100 most common homophone tuples?
    # print(stats["Homophone_Pair"].value_counts()[0:100])

    # 2. Which tokens are the most frequent?
    # print(stats["Homophone"].value_counts()[0:100])

    #3. How many instances are there of each element of tuple?

    #Get count by homophone tuple and each individual homophone.
    stats.groupby(["Homophone_Tuple", "Homophone"]).size()
    group = stats.groupby(["Homophone_Tuple", "Homophone"]).size().reset_index(name="Count")
    group["Homophone_Total"] = group.groupby("Homophone_Tuple")["Count"].transform(
        "sum")
    # print(group.sort_values("Homophone_Total", ascending=False)[50:100])

    # Write to file. Counts are written as strings!
    # group.to_csv("homophone_stats.csv", header=True, encoding="utf-8", sep="\t")

    # 4. Percentages of frequencies of tokens by homophone pair.
    group["Percentage"] = (group["Count"] / group["Homophone_Total"]) *100
    # print(group[50:100])

    # 5. Which Homophones and homophone tuples do not appear  in data?
    # unseen_tuples = []
    # for tuple in hphones.__homophone_tuples__:
    #     tup = " ".join(hphone for hphone in tuple)
    #     if stats['Homophone_Tuple'].str.contains(tup).any() == False:
    #         unseen_tuples.append(tup)
    # print(unseen_tuples)
    # print(len(unseen_tuples))
    # Compare with unique tokens. Check if sum adds to 420 (Total in list).
    # print(len(stats.Homophone_Tuple.unique()))


    # 6. Homophone tags of homophones!



if __name__ == "__main__":
    main()