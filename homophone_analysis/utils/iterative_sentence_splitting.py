#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2017


'''
Perform sentence splitting while iterating over fixed-sized chunks of text.
'''


import re
import sys
from typing import Iterator, TextIO
from nltk.tokenize.util import string_span_tokenize

import nltk.data

# Get an English sentence splitter.
punkt = nltk.data.load('tokenizers/punkt/english.pickle')


def main():
    '''
    Run as script: Read STDIN, write to STDOUT.
    '''
    sentences = iter_sentences(open("phrases.en", "r"))
    for sentence in sentences:
        # Remove any superfluous whitespace (especially newlines).
        # sentence = re.sub(r'\s+', ' ', sentence)
        print(sentence)


def iter_sentences(stream: TextIO) -> Iterator[str]:
    '''
    Iterate over sentences from a text stream.
    '''
    remainder = ''
    for chunk in iter_chunks(stream):
        # Add remainder from the previous chunk.
        chunk = remainder + chunk
        *spans, last = string_span_tokenize(chunk, "\n")
        for start, end in spans:
            yield chunk[start:end]
        # Keep the last sentence -- it might be continued in the next chunk.
        remainder = chunk[last[0]:]
    # Remember to yield the very last remainder.
    if remainder:
        yield remainder


def iter_chunks(stream: TextIO, chunksize: int = 1000) -> Iterator[str]:
    '''
    Iterate over chunks of fixed size.
    '''
    while True:
        chunk = stream.read(chunksize)
        if not chunk:
            # End of file reached.
            break
        yield chunk




if __name__ == '__main__':
    main()
