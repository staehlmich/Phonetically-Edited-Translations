#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

def lowercase_file(input_file, output_file):
    """
    Function to lowercase file. Lowercase file is needed for MFA input.
    @param input_file: file to generate phoneme dictionary.
    @param output_file: lowercased file
    @return:
    """
    with open(input_file, "r", encoding="utf-8") as infile, \
            open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            outfile.write(line.lower())

def main():
    # path = "/home/user/staehli/master_thesis/data/MuST-C/en-de/data/train/txt/"
    outpath = "/home/user/staehli/master_thesis/homophone_analysis/mfa_input/"
    # train_en = path+"train.en"
    # train_de = path+"train.de"
    # train_lc_en = outpath+"train.lc.en"
    # train_lc_de = outpath+"train.lc.de"
    # lowercase_file(train_en, train_lc_en)
    # lowercase_file(train_de, train_lc_de)
    test_source = "/home/user/staehli/master_thesis/data/MuST-C/en-de/data/train/txt/train.en"
    lowercase_file(test_source, outpath+"train.lc.en")

if __name__ == "__main__":
    main()