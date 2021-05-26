#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

"""Script to format output of moses phrase table to train with MFA."""
def main():
    phrase_table_path = "/home/user/staehli/master_thesis/homophone_analysis/moses_experiments/model/extract.sorted"

    with open(phrase_table_path, "r", encoding="utf-8") as infile:
        phrases = []
        for line in infile:
            line = line.split("|||")[:2]
            phrases.append((line[1], line[0]))
        phrases = set(phrases)
        with open("phrases.en", "w", encoding="utf-8") as phrases_source, open("phrases.de", "w", encoding="utf-8") as phrases_target:
            for elem in phrases:
                phrases_source.write(elem[0]+"\n")
                phrases_target.write(elem[1]+"\n")



if __name__ == "__main__":
    main()