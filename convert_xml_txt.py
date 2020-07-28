#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Michael Staehli
# original idea: https://stackoverflow.com/questions/2941264/how-to-convert-xml-files-to-text-files/2941274#2941274

import sys
import os
import xml.etree.ElementTree as ET
def Readthexml(f):
    """Read the file from the argument list and dump the title contents and keywords"""
    tree = ET.parse(f)
    tags = ["seg"]

    with open(f.rstrip(".xml"), "w") as outfile:
        for child in tree.iter():
            if child.tag in tags:
                outfile.write(child.text.lstrip()+"\n")
    return True

def read_training(f):
    """
    Read the training file from the argument list and write text without
    metadata to file.
    """

    with open(f, "r", encoding="u8") as infile:
        with open(f[:-8]+f[-3:], "w") as outfile:
            for line in infile:
                if line.startswith("<") == False:
                    outfile.write(line)
    return True

def main(argv=None):
    if argv is None:
        argv = sys.argv
        args = argv[1:]
        for arg in args:
            if os.path.exists(arg):
                for file in os.listdir(arg):
                    if file.endswith(".xml"):
                        filepath = arg+"/"+file
                        Readthexml(filepath)
                    if file.startswith("corpus"):
                        filepath = arg+file
                        read_training(filepath)

if __name__ == "__main__":
    main()