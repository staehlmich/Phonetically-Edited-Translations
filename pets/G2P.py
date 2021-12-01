#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

import argparse
import re

import epitran
from g2p_en import G2p
from ipapy.ipastring import IPAString

"""Script to phonetize source phrases of filtered phrase table.
Phonetized tokens are not separated by whitespace.
Install epitran with lex_lookup from: https://github.com/dmort27/epitran
Next step: Run PETs."""

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
    if phonebet == "IPA":
        # Remove suprasegemental features with ipapy.
        # Suprasegmental features not supported with wordkit in PETS.
        return no_supra(phonemizer.transliterate(seq))
    if phonebet == "ARPA":
        # Remove added apostrophes by phonemizer.
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
    Phonetized tokens are not separated by whitespace.
    """
    with open(table_path, "r", encoding="utf-8") as infile, \
        open(table_path[:-5]+f"ph.{phonebet.lower()}.en-de", "w", encoding="utf-8") as out:
        phonemizer = set_g2p(phonebet)
        for line in infile:
            src, trg = (elem for elem in line.split("|||"))
            src_phon = g2p(src, phonebet, phonemizer)
            out.write(sep.join([src.strip(" "), src_phon, trg.strip(" ")]))

def main():
    parser = argparse.ArgumentParser(description="Script to phonetize source phrases of filtered phrase table.")
    parser.add_argument("tablepath", help="Path to filtered phrase table file.")
    parser.add_argument("-p", "--phonebet", help="Choose phonetic alphabet: IPA or ARPA.", default="IPA")
    args = parser.parse_args()
    table_g2p(args.tablepath, args.phonebet)

if __name__ == "__main__":
    main()