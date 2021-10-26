#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

import re
import argparse
# from dp.phonemizer import Phonemizer
from g2p_en import G2p
import epitran
from ipapy.ipastring import IPAString

def no_supra(ipa_string:str) -> str:
    """
    Function to remove suprasegmental symbols from IPA phonetic string.
    @param ipa_string: phrase in IPA notation.
    @return: Phonetic IPA string without suprasegmental symbols.
    """
    phonemes = []
    for phon in ipa_string:
        features = IPAString(unicode_string=phon,
                             single_char_parsing=True, ignore=True)
        rest = all(
            [x.is_diacritic or x.is_suprasegmental for x in features])
        if rest == False:
            phonemes.append(phon)
    return "".join(phon for phon in phonemes)

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
        #Remove suprasegemental features with ipapy.
        return no_supra(phonemizer.transliterate(seq))
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
            #TODO: strip whitespace at end of src_phon (IPA)
            out.write(sep.join([src.strip(" "), src_phon, trg.strip(" ")]))

def main():

    # table_g2p("phrases.filtered4.en-de", "ARPA")
    table_g2p("phrases.filtered5.en-de", "IPA")


if __name__ == "__main__":
    main()