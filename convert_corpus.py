#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Michael Staehli
# original idea: https://stackoverflow.com/questions/2941264/how-to-convert-xml-files-to-text-files/2941274#2941274

import sys
import os
import xml.etree.ElementTree as ET

def main(argv=None):
    if argv is None:
        argv = sys.argv
        args = argv[1:]
        for arg in args:
            if os.path.exists(arg):
                for file in os.listdir(arg):
                    if file.startswith("corpus"):
                        filepath = arg+"/"+file
                        print(filepath)

if __name__ == "__main__":
    main()