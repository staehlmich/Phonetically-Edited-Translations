#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

def main():
    counter = 0
    sents_id = []
    with open("data/MuST-C/test.tc.de", "r", encoding="utf-8") as source, \
            open("results.tc.txt", "r", encoding="utf-8") as target:
                for line in zip(source, target):
                    counter += 1
                    src = line[0].split()
                    trg = line[1].split()
                    if len(src) != len(trg):
                        sents_id.append(counter)

    print(len(sents_id))
    print(sents_id)


if __name__ == "__main__":
    main()