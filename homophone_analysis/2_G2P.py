#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

from typing import TextIO
import re
import argparse
import pandas as pd
import wordkit
from wordkit.corpora.corpora.cmudict import CMU_2IPA
from wordkit.corpora import reader
from g2p_en import G2p

class CMUReader(object):
    """
    Wrapper class to call wordkit.corpora.corpora.cmudict, with some
    minor modifications: phoneme strings are saved as Arpabet symbols.
    Adapted methods from wordkit: self._open and self._cmu_to_cmu.
    The second method is originally called self._cmu_to_ipa.
    """
    def __init__(self, path:str):
        self._fields = ('orthography', 'phonology')
        self.df = reader(path, self._fields, language="eng", opener=self._open, preprocessors={"phonology": self._cmu_to_cmu})

    def _open(self, path, **kwargs):
        """Open a file for reading."""
        df = []
        brackets = re.compile(r"\(\d\)")
        #Set encoding for cmu-file from official site.
        #http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/
        for line in open(path, encoding="ISO-8859-1"):
            line = line.split("\t")[0]
            word, *rest = line.strip().split()
            word = brackets.sub("", word).lower()
            df.append({"orthography": word, "phonology": rest})
        return pd.DataFrame(df)

    def _cmu_to_cmu(self, phonemes):
        """Hack-method to avoid converting Arpabet symbols to IPA."""
        # Set values of CMU_2IPA to Arpabet keys.
        for key in CMU_2IPA:
            CMU_2IPA[key] = key
        return tuple([CMU_2IPA[p] for p in phonemes])

    def word_g2p(self, word: str) -> str:
        """
        Helper method to retrieve phonetic representation from self.df.
        @param word: Grapheme representation of word.
        @return: Phonetic reprsentation in Arpabet symbols.
        """
        phonemes = self.df.loc[self.df['orthography'] == word]["phonology"].item()
        return " ".join(elem for elem in phonemes)

def table_g2p(table_path: str, sep="\t") -> TextIO:
    """
    Function that takes a filtered moses phrase table and outputs a file
    that also contains a phonetically converte source phrase.
    @param table_path: fitlered phrase table file.
    Line contains: (source_phrase ||| target_phrase1/.../target_phraseN)
    @param sep: symbol to separate source, phonemes, translations.
    @return: File contains lines with:
    (source_phrase, phonetized phrase, target_phrases)
    """
    with open(table_path, "r", encoding="utf-8") as infile, \
        open(table_path[:-5]+"ph.en-de", "w", encoding="utf-8") as out:
        g2p = G2p()
        for line in infile:
            src, trg = (elem for elem in line.split("|||"))
            src_phon = " ".join(g2p(src))
            out.write(sep.join([src.strip(" "), src_phon, trg.strip(" ")]))

def main():
    #Read CMU pronunciation dictionary with wordkit.
    # cmu = CMUReader("cmudict-0.7b.txt")

    table_g2p("phrases.filtered3.en-de")


if __name__ == "__main__":
    main()