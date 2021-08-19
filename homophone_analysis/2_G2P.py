#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

import re
import argparse
from g2p_en import G2p
# from dp.phonemizer import Phonemizer
import epitran


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

def set_g2p(phonebet: str):
    """
    Helper function to instantiate G2P instance.
    @param phonebet: If "IPA" Epitran is set as G2P phonemizer.
    If "ARPA" G2p is set as G2P phonemizer.
    @return:
    """
    if phonebet == "IPA":
        return epitran.Epitran("eng-Latn")
    if phonebet == "ARPA":
        return G2p()

def g2p(seq: str, phonebet: str, phonemizer) -> str:
    """
    Function to phonetize a string.
    @param seq: String to be phonetized.
    @param phonebet: If "IPA" Epitran is set as G2P phonemizer.
    @param phonemizer: Instantiated G2P madel
    If "ARPA" G2p is set as G2P phonemizer.
    @return:
    """
    #TODO: Fix whitespaces at the end of strings!
    if phonebet == "IPA":
        return phonemizer.transliterate(seq)
    if phonebet == "ARPA":
        #Remove added apostrophes by phonemizer.
        return " ".join(phonemizer(seq)).replace(" '", "")

def table_g2p(table_path: str, phonebet: str, sep="\t"):
    """
    Function that takes a filtered moses phrase table and outputs a file
    that also contains a phonetically converte source phrase.
    @param table_path: fitlered phrase table file.
    Line contains: (source_phrase ||| target_phrase1/.../target_phraseN)
    @param phonebet: Phonetization with IPA or ARPA.
    @param sep: symbol to separate source, phonemes, translations.
    @return: File contains lines with:
    (source_phrase, phonetized phrase, target_phrases)
    """
    with open(table_path, "r", encoding="utf-8") as infile, \
        open(table_path[:-5]+f"ph.{phonebet.lower()}.en-de", "w", encoding="utf-8") as out:
        #TODO: instantiate G2P package before loop, to avoid prints.
        phonemizer = set_g2p(phonebet)
        for line in infile:
            src, trg = (elem for elem in line.split("|||"))
            src_phon = g2p(src, phonebet, phonemizer)
            out.write(sep.join([src.strip(" "), src_phon, trg.strip(" ")]))

def main():

    table_g2p("phrases.filtered4.en-de", "ARPA")
    # table_g2p("phrases.filtered3.en-de", "IPA")


if __name__ == "__main__":
    main()