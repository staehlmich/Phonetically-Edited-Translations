#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

from typing import Iterator, Iterable, Generator
import re
import string
import timeit
from collections import OrderedDict, defaultdict, Counter
from collections.abc import MutableMapping
import numpy as np
import nltk
import itertools
from itertools import islice, groupby
from g2p_en import G2p
import Levenshtein
from nltk.metrics import edit_distance

class ArpabetChar(str):
    """
    Class that turn string into an Arpabet character.
    http://www.speech.cs.cmu.edu/cgi-bin/cmudict
    """
    def __init__(self, chars: list):
        self._chars = chars

    def __repr__(self):
        return "".join(char for char in self._chars)

    def __str__(self):
        return "".join(char for char in self._chars)

    def __eq__(self, other):
        if self._chars == other._chars:
            return True
        else:
            return False

    def __len__(self):
        return len(self._chars)

    def __setitem__(self, key, value):
        self._chars[key] = value

    def __getitem__(self, item):
        return ArpabetChar([self._chars[item]])

    def __contains__(self, item):
        if item in self._chars:
            return True
        else:
            return False

    def __iter__(self):
        for char in self._chars:
            yield ArpabetChar([char])

    def __add__(self, other):
        added_char = [char for char in self._chars]
        for char in other._chars:
            added_char.append(char)
        return ArpabetChar([added_char])

    # def __iadd__(self, other):
    #     for char in other._chars:
    #         self._chars.append(char)

class SuperString(object):
    """
    Class that saves phonetic string, grapheme string and translation of
    token or phrase.
    """
    def __init__(self, grapheme: str, phoneme: str, translation: str):
        self.grapheme = grapheme
        self.phoneme = phoneme
        #Phoneme string without whitespaces.
        self.joined_phoneme = self.join_phon()
        self.translation = translation

    def __repr__(self):
        return f'<{self.__class__.__name__}: {(self.grapheme, self.phoneme, self.translation)}>'

    def __eq__(self, other):
        if [self.grapheme, self.phoneme, self.translation] == \
                [other.grapheme, other.phoneme, other.translation]:
            return True
        else:
            return False
    def join_phon(self) -> str:
        """
        Method to remove whitespaces from phonetic string.
        @param phrase: self.phoneme
        @return:
        """
        if " " in self.phoneme:
            return "".join(elem for elem in self.phoneme.split())
        else:
            return self.phoneme

class PhonDict(MutableMapping):
    """
    Class to read a phonetic dictionary. Filename is phonetized vocabulary
    file or phrase table, with the following lines:
    grapheme string, phoneme string, translation.
    """
    def __init__(self, filename):
        self._filename = filename
        self._set_vocab()

    def __repr__(self):
        return str(self._store)

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def values(self):
        return self._store.values()

    def keys(self):
        return self._store.keys()

    def items(self):
        return self._store.items()

    def _extract_phon_dictionary(self):
        """
        Method that reads self._filename (phontized vocabulary or phrase table).
        Yields SuperString obects with: grapheme string, phonetic string, translation.
        @return:
        """
        with open(self._filename, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.rstrip().split("\t")
                if len(line) != 1:
                    yield SuperString(line[0], line[1], "None")

    def _set_vocab(self):
        """
        Method that sets dict-like phonetic table. Phonetic table contains
        class SuperString objects. Key is grapheme representation of token/phrase.
        @return:
        """
        self._store = {token.grapheme:token for token in self._extract_phon_dictionary()}

    def get_homophones(self) -> dict:
        """
        Method to group SuperString by .phoneme attribute. Separate return,
        because function is rarely used.
        @return: SuperString.phoneme: [SuperString.phoneme1, ..., n]
        """
        d = defaultdict(list)
        homophones = OrderedDict()
        #Reverse keys (graphemes) with values (phonemes).
        for key, value in self._store.items():
            d[value.phoneme].append(value.grapheme)
        #Add only homophones to new dictionary.
        for phoneme, grapheme in d.items():
            if len(grapheme) > 1:
                homophones[phoneme] = grapheme
        return homophones

class PhonPhrases(PhonDict):
    """
    Subclass of PhonDict to read phonetic representations of phrases.
    """
    def __init__(self, vocab_file):
        super().__init__(vocab_file)

    def _extract_phon_dictionary(self):
        """
        Method that reads self._filename (phontized vocabulary or phrase table).
        Yields SuperString obects with: grapheme string, phonetic string, translation.
        @return:
        """
        with open(self._filename, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.rstrip().split("\t")
                yield SuperString(line[0], line[1], line[2])

    def _phrases_by_length(self, n: int, max_dist=3):
        """
        Method to yield all elements from self._store than are in range
        n+-max_dist.
        @param n: Length of searchphrase.
        @param max_dist: maximum edit operations in phonetic_levenshtein
        @return:
        """
        r = range(n-max_dist, n+max_dist+1)
        for elem in self._store.values():
            if len(elem.joined_phoneme) in r:
            # if max(1, n-max_dist) <= len(elem.joined_phoneme) <= n+max_dist:
                yield elem

    def phrase_to_phon(self, phrase:str) -> str:
        #TODO: Adapt for CMU!
        """
        Method that looks up phonetic representation in self._store.
        If phrase not in self._store, resort to Epitran as slower method.
        @param phrase: phrase as grapheme string.
        @return: IPA representation of phrase.
        """
        try:
            return self._store[phrase].phoneme
        except KeyError:
            g2p = G2p()
            return g2p(phrase)

    def phonetic_levenshtein(self, search: str, max_dist=3) -> Iterator:
        """
        Function to retrieve all phonetic phrases from self.vocab (phonetic
        phrase table) with a levenshtein distance <= max_dist.
        @param search: Search phrase.
        @param max_dist: Highest number of edit operations.
        @return: yields SuperString object.
        """
        search_phon = self.phrase_to_phon(search)
        joined_search = "".join(elem for elem in search_phon.split())
        #Only look at phrases of length +- max_dist
        for superstring in self._phrases_by_length(len(joined_search)):
            #Filter phrases by same number of tokens.
            #TODO: Save this format in superstring? Or as method Makes if very slow?
            #Look up levenshtein distance with joined phonetized phrase.
            # TODO: optimal distance?
            if Levenshtein.distance(joined_search, superstring.joined_phoneme) <= max_dist:
                # Filter out longest substring, so that not additional tokens.
                # Check grapheme tokens, to avoid different phonetic representations of longer strings.
                # if self.longest_token(search,superstring.grapheme) == False:
                #TODO: precision filters on results in subsequent function!
                #if len(superstring.grapheme) > 3:
                #Speedup/precision: translation to higher number of tokens is unlikely?
                if superstring.grapheme.count(" ") <= search.count(" "):
                    # Check if search phrase and otuput phrase are not the same.
                    if search != superstring.grapheme:
                        yield superstring

    @staticmethod
    def longest_token(phrase1: str, phrase2: str) -> bool:
        """
        Helper function (calculate levenshtein distance) to check if to phrases share a common token.
        @param phrase1: Source phrase from phon_table.
        @param phrase2: Other source phrase from phon_table.
        @return:
        """
        matches1 = any(elem for elem in phrase1.split(" ") if elem in phrase2)
        matches2 = any(elem for elem in phrase2.split(" ") if elem in phrase1)
        if matches1 or matches2:
            return True
        return False

    @staticmethod
    def weighted_distance(phrase1, phrase2, costs=(1, 1, 1)):
        """
        Levenshtein distance with modified costs using python-Levenshtein
        package base ond C++.
        @param phrase1: phrase in grapheme or phonetic representation.
        @param phrase2: phrase in grapheme or phnetic representation
        @param costs: a tuple or a list with three integers (d, i, s)
        @return:
        """
        c = Counter(elem[0] for elem in Levenshtein.editops(phrase1, phrase2))
        d = c["insert"]*costs[0]
        i = c["delete"]*costs[1]
        s = c["replace"]*costs[2]
        return d+i+s

class Candidates(object):
    """
    # I don't want to use append (slow)
    # I want to have source and target candidates
    # I don't want to save the candidates in class instance
    # I want to save the index of target or source, to compare later after levdist.
    # For the last item, I could use this: sorted(l, key = lambda x: test_ex.index(x)) (compare searchphrase to match of levdist).
    """
    def __init__(self, source_sent: str, target_sent: str):

        self._source_sent = source_sent
        self._target_sent = target_sent

    def _yield_candidates(self, phon_table):
        """
        Method to check which ngrams don't have a translation in phon_table.
        @param phon_table: Instance of PhonPhrases.
        @return: src_cands = list with source phrases that have no translation in phon_table.
        trg_cands = list of translation phrases, that can't be aligned to src_cands.
        """
        #TODO: code repetition
        source_grams = list(" ".join(elem) for elem in nltk.everygrams(self._source_sent.split(), max_len=3))
        #Use set for faster lookups.
        target_grams = {" ".join(elem) for elem in nltk.everygrams(self._target_sent.split(), max_len=3)}
        non_cands = [phon_table[src] for src in set(source_grams).intersection(phon_table.keys()) if phon_table[src].translation in target_grams]
        src_non_cands = {elem.grapheme for elem in non_cands}
        trg_non_cands = {elem.translation for elem in non_cands}
        # src_non_cands = {phon_table[src].grapheme for src in source_grams if src in phon_table.keys() if phon_table[src].translation in target_grams}
        # trg_non_cands = {phon_table[src].translation for src in source_grams if src in phon_table.keys() if phon_table[src].translation in target_grams}
        src_cands = list({src for src in source_grams if src not in src_non_cands if len(src) > 3})
        trg_cands = {trg for trg in target_grams if trg not in trg_non_cands if len(trg) > 3}
        return src_cands, trg_cands

    def phon_edit_translations(self, phon_table) -> Iterator:
        """
        Method to yield all translations of phonetically similar strings of
        ngrams in src_cands.
        @param phon_table: Instance of PhonPhrases.
        @return:
        """
        src_cands, trg_cands = self._yield_candidates(phon_table)
        # print(len(src_cands), len(trg_cands))
        for src_cand in src_cands:
            # Simphones are phonetically similar strings (lev-dist <=3)
            for simphone in phon_table.phonetic_levenshtein(src_cand,max_dist=3):
                if simphone.translation in trg_cands:
                    #Filter phrases that add tokens.
                    if phon_table.longest_token(src_cand, simphone.grapheme) == False:
                        #Filter by index position.
                        if abs(self._source_sent.split().index(src_cand.split()[0])-self._target_sent.split().index(simphone.translation.split()[0])) <=3:
                            yield (src_cand, simphone.grapheme, simphone.translation)

class FileReader(object):
    """
    Class to open and read files. Returns lines.
    """
    def __init__(self, filepath: str, lang:str,*args, mode=""):
        self._filepath = filepath
        self._lang = lang
        self._mode = mode
        self._args = None
        if self._mode == "":
            raise TypeError("Choose mode: line, token, no_punct")
        if args:
            self._args = args[0]

    def _line_iter(self) -> Iterator:
        """
        Method to iterate over lines.
        @return:
        """
        with open(self._filepath, "r", encoding="utf-8") as infile:
            if self._args:
                for line in islice(infile, self._args):
                    yield line
            else:
                for line in infile:
                    yield line

    def get_lines(self):
        """
        Method to yield lines, according to parameter mode.
        @return:
        """
        tokenizer = MosesTokenizer(self._lang)
        for line in self._line_iter():
            if self._mode == "line":
                yield line
            elif self._mode == "token":
                yield " ".join(elem for elem in tokenizer.tokenize(line))
            elif self._mode == "no_punct":
                yield " ".join(elem for elem in tokenizer.tokenize(line) if elem not in string.punctuation)

def foo():
    """
    To timeit methods.
    @return:
    """
    char1 = ArpabetChar(["AH0"])
    char3 = ArpabetChar(["AE1"])
    print(f"Levenshtein distance {char1} and {char3}:{minimum_edit_distance(char1, char3)}")

def foo2():
    """
    To timeit 2 method.
    @return:
    """
    char1 = ArpabetChar(["AH0"])
    char3 = ArpabetChar(["AE1"])
    print(f"NLTK levenshtein distance{char1} and {char3}: {edit_distance(char1, char3)}")

def main():

    ### 1. Phonetize vocabulary and phrase table###
    ### 1. Phonetize vocabulary and phrase table###
    ### 2. Get phonetic dictionary and homophones ###
    # phon_dic = PhonDict("phrases.filtered3.ph.en-de")
    # homophones = phon_dic.get_homophones()
    # print("Length of PhonDict:", len(phon_dic))
    # print("Numer of homophone types:", len(homophones))
    # print(phon_dic["weak"])

    # src, src_phon, trans = (elem for elem in "a boy 	AH0   B OY1	 a boy/ein junge/einen jungen".split("\t"))
    # super = SuperString(src, src_phon, trans)
    # print(super.phoneme)
    #
    # char1 = ArpabetChar(["AH0"])
    # char2 = ArpabetChar(["AH1"])
    # char3 = ArpabetChar(["AE1"])
    # print("Indexing:", char1[0])
    # print(f"Length of {char1}: {len(char1)}")
    # print(f"Length of {char2}: {len(char2)}")
    #
    # print(f"Levenshtein distance {char1} and {char2}:{Levenshtein.distance(char1, char2)}")
    # print(f"Levenshtein distance {char1} and {char3}:{Levenshtein.distance(char1, char3)}")
    #
    # char4 = ArpabetChar(["AH0", "B", "OY1"])
    # new_char = char4[2]
    # print(new_char[0], len(new_char))
    # print(len(str(char1)))
    # print(f"NLTK levenshtein distance {char1} and {char2}: {edit_distance(char1, char2)}")
    # print(f"NLTK levenshtein distance {char1} and {char3}: {edit_distance(char1, char3)}")

    # print(timeit.timeit("foo()", globals=globals(), number=5))
    # print(timeit.timeit("foo2()", globals=globals(), number=5))


    ### 2. Get phonetic dictionary and homophones ###

    ### 3. Phonetize phrase table ###
    phon_table = PhonPhrases("phrases.filtered3.ph.en-de")
    print("Number of phrases:", len(phon_table))
    #Overview: Count phrases by number of tokens.
    token_lengths = Counter(
        map(lambda x: x.grapheme.count(" "), phon_table.values()))
    print(phon_table["written"])


    # homophone_phrases = phon_table.get_homophones()
    # print("Numer of homophones in phrase table:", len(homophone_phrases))
    # counter = 0
    # for key in homophone_phrases:
    #     if key.count(" ") >= 1:
    #         counter +=1
    # print("Numer of homophone phrases:", counter)
    # print(len(homophones), len(homophone_phrases))
    ### 3. Phonetize phrase table ###

    ### 4. Find phrases with smallest levenshtein distance. ###
    #Test with simple phrase#

    # search = "ɪz ɹɪtən"
    # search2 ="ɪn ɪɡzæktli"
    # search3 = "in"
    # search4 = "lɪvɪŋ kɹit͡ʃɹ̩"

    # s1 = SuperString('britain', 'bɹɪtən', 'großbritannien')
    # s2 = SuperString('britain', 'bɹɪtən', 'großbritannien')
    # s3 = SuperString('in urban', 'ɪn ɹ̩bən', 'in städtischen')
    # l = [s1]
    # print(s2 in l)
    # print(s3 in l)
    # print(type(s1))

    # similar = list(phon_table.phonetic_levenshtein("is written", max_dist=3))
    # for item in similar:
    #     print(item.grapheme, "|||", item.translation)
    # print("\n")

    # print(len(similar))
    # print(timeit.timeit("foo2()", globals=globals(), number=5))
    # print(phon_table["is written"], phon_table["wharton"])
    # ex1 = phon_table["is written"]
    # ex2 = phon_table["if britain"]
    # print(ex1, ex2)

    #Test with simple phrase#

    #Test with sentences#
    # Errors: homograph "letters", homophone "set"
    source_ex = "In fact every living creature is written in exactly the same set of letters and the same code".lower()
    test_ex = "Und in jedem Lebewesen ist es sozusagen aus Großbritannien genau mit den Buchstaben am selben Code".lower()
    target_ex = "Und zwar jedes Lebewesen verwendet die exakt gleichen Buchstaben und denselben Code".lower()
    source_ex2 = "I filmed in war zones difficult and dangerous".lower()
    test_ex2 = "Ich fühlte mich in den Kriegsgebieten schwer und gefährlich".lower()
    source_ex3 = "the captain waved me over"
    test_ex3 = "der kapitän wartete darauf"
    # source_ex4 = "I was the second volunteer on the scene so there was a pretty good chance I was going to get in".lower()
    # test_ex4 = "Ich war die zweite freiwillige Freiwillige am selben Ort also gab es eine sehr gute Chance das zu bekommen".lower()

    # candidates = Candidates(source_ex, test_ex)
    # for match2 in candidates.phon_edit_translations(phon_table):
    #     print(match2)
    # print(source_ex)
    # print(test_ex)

    # source_grams = list(" ".join(elem) for elem in nltk.everygrams(source_ex.split(), max_len=3))
    # target_grams = list(" ".join(elem) for elem in nltk.everygrams(target_ex.split(), max_len=3))
    # test_grams = list(" ".join(elem) for elem in nltk.everygrams(test_ex.split(), max_len=3))
    # non_cands = [phon_table[src] for src in set(source_grams).intersection(phon_table.keys()) if phon_table[src].translation in target_grams]
    # non_cands_test = [phon_table[src] for src in set(source_grams).intersection(phon_table.keys()) if phon_table[src].translation in test_grams]

    # print(phon_table["written"])
    # print(len(phon_table["written"].phoneme))
    # print(d["water"])

    #Phon-Table stats (phrases by number of tokens).
    # token_lengths = Counter(map(lambda x: x.grapheme.count(" "), phon_table.values()))
    #Phon-table by number of characters of phrases (without counting whitespace).
    # char_lengths = Counter(map(lambda x: len(x.phoneme)-x.phoneme.count(" "), phon_table.values()))
    # print(char_lengths)
    # print(char_lengths[5])

    # neighborhood_ex = list(phon_table.phonetic_levenshtein("written", max_dist=1))
    # print(neighborhood_ex)
    # print(len(neighborhood_ex))

    #TODO: Testing
    # print(counter[1]+counter[2]+counter[3]+counter[4]+counter[5]+counter[6])
    # sims = {elem.grapheme:list(phon_table.phonetic_levenshtein(elem.grapheme))for elem in phon_table._phrases_by_length(5)}
    # print(sims["water"])
    # print(timeit.timeit("foo()", globals=globals(), number=5))
    # print("\n")
    # print(timeit.timeit("foo2()", globals=globals(), number=1))

    #Test with sentences#
    ### 4. Find phrases with smallest levenshtein distance. ###


    ###5. Test on files ###

    # src = FileReader("tst-COMMON.en", "en", 10, mode="no_punct")
    # tst = FileReader("trans.tok.txt", "de", 10, mode="no_punct")
    # for source, hyp in zip(src.get_lines(), tst.get_lines()):
    #     pets = Candidates(source.lower(), hyp.lower())
    #     for pet in pets.phon_edit_translations(phon_table):
    #         print(pet)
    #     print(source)
    #     print(hyp)
    #     print("\n")
    # ###5. Test on files ###

if __name__ == "__main__":
    main()