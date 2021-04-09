#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

def main():
    with open("test.en-de", "w", encoding="utf-8") as outfile, \
            open("data/MuST-C/test.tc.en", "r", encoding="utf-8") as source, \
            open("results.tc.txt", "r", encoding="utf-8") as target:
                for line in zip(source, target):
                    src_line = line[0].rstrip("\n")
                    trg_line = line[1]
                    outfile.write(src_line+" ||| "+trg_line)

if __name__ == "__main__":
    main()