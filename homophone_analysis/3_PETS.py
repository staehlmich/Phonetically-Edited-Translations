#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

from typing import Iterator, Iterable, Generator
import re
import string
import timeit
import cProfile
from collections import OrderedDict, defaultdict, Counter
from collections.abc import MutableMapping
import numpy as np
import nltk
import itertools
from itertools import islice, groupby
from g2p_en import G2p
import Levenshtein
from nltk.metrics import edit_distance
from wordkit.features import CVTransformer, OneHotPhonemeExtractor
from wordkit.corpora.corpora import cmudict
from scipy.spatial import distance

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
                    if line[0].count(" ") == 0:
                        yield SuperString(line[0], line[1], line[2])

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

    #TODO: parameter CMU/IPA
    def phrase_to_phon(self, phrase:str) -> str:
        """
        Method that looks up phonetic representation in self._store.
        If phrase not in self._store, resort to Epitran/g2p as slower method.
        @param phrase: phrase as grapheme string.
        @return: IPA representation of phrase.
        """
        try:
            return self._store[phrase].phoneme
        except KeyError:
            g2p = G2p()
            return " ".join(g2p(phrase))

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
        #Count the number of stresses.
        # num_stresses = sum(c.isdigit() for c in joined_search)
        # Only look at phrases of length +- max_dist
        for superstring in self._phrases_by_length(len(joined_search)):
            #Filter phrases by same number of tokens.
            #TODO: Save this format in superstring? Or as method Makes if very slow?
            #Look up levenshtein distance with joined phonetized phrase.
            # TODO: optimal distance?
            # if Levenshtein.distance(joined_search,
            #                         superstring.joined_phoneme) <= max_dist:
            if Levenshtein.distance(joined_search, superstring.joined_phoneme) <= round(len(joined_search)/3):
            # if edit_distance(ArpabetChar(search_phon.split(" ")), ArpabetChar(superstring.phoneme.split(" ")), substitution_cost=2) <= max_dist:
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
    Class that uses a phonetic table (PhonPhrases) to search which phrases
    from source_sent have no translations in target_sent.
    """
    def __init__(self, source_sent: str, target_sent: str):
        #TODO: faster if phon_table as parameter?
        self._source_sent = source_sent
        self._target_sent = target_sent

    def everygrams(self, sent: str, length=3) -> Iterator:
        """
        Helper method to return ngrams of order n up to to length.
        @param sent: Source or target sentence.
        @param length: Highest ngram order.
        @return:
        """
        return (" ".join(elem) for elem in nltk.everygrams(sent.split(), max_len=length))

    def _search_translation(self, superstring, trg_sent: str):
        """
        Helper method to check if any of the translations of a superstring
        are in the target sentence. Return None or the highest scored
        translation.
        @param superstring: instance of class Superstring.
        @param trg_sent: target sentence.
        @return: [] or highest scored translation.
        """
        hits = []
        for trans in superstring.translation.split("/"):
            hit = re.search(rf"\b{trans}\b", trg_sent)
            if hit != None:
                hits.append(hit.group(0))
        if hits == []:
            return []
        else:
            return hits[0]

    def _search_candidates(self, phon_table, mode="cand"):
        """
        Method to check which ngrams don't have a translation in phon_table.
        @param phon_table: Instance of PhonPhrases.
        @param mode: If mode == "cand" yield source candidate phrases.
        If mode == "non_cand" yield target non candidate phrases.
        @return: generator of source candidate strings or target non
        candidate strings.
        """
        src_grams = [" ".join(elem) for elem in
                     nltk.everygrams(self._source_sent.split(), max_len=3)]
        keys = set(phon_table.keys())
        for src in src_grams:
            if src in keys:
                value = phon_table[src]
                if re.search(rf"\b{src}\b", self._source_sent) != None:
                    trans_match = self._search_translation(value, self._target_sent)
                    if trans_match == []:
                        if mode == "cand":
                            yield value.grapheme
                    else:
                        if mode == "non_cand":
                            yield trans_match

    def _filter_candidates(self, phon_table):
        """
        Method to filter candidates. Keep only phrases, when all of its tokens
        do not have a translation in self._target_sent.
        @return:
        """
        candidates = list(self._search_candidates(phon_table))
        new_candidates = [cand for cand in candidates if all(elem in candidates for elem in cand.split()) == True]

        return new_candidates

    def pets(self, phon_table) -> Iterator:
        """
        Method to yield all translations of phonetically similar strings of
        ngrams in src_cands.
        @param phon_table: Instance of PhonPhrases.
        @return:
        """
        #TODO: save candidates and non candidates in self?
        #TODO phon_table as class parameter?
        src_cands = self._filter_candidates(phon_table)
        non_cands = list(self._search_candidates(phon_table, mode="non_cand"))
        for src_cand in src_cands:
            # Search phonetically similar strings (lev-dist <=3)
            src_cand_phon = phon_table.phrase_to_phon(src_cand)
            #Additional filter with cosine similarity.
            simphones = list(phon_table.phonetic_levenshtein(src_cand,max_dist=3))
            #Hack to avoid empty grid in CVTransformer.
            if len(simphones) > 10:
                phonsims = PhonSimStrings(simphones)
                for simphone in phonsims.most_similar(src_cand_phon, sim=0.6):
                    #Check if translations in target sentence.
                    for trans in simphone.translation.split("/"):
                        if re.search(rf"\b{trans}\b", self._target_sent) != None:
                            #Check if trans already has a gold translation.
                            if trans not in non_cands:
                                yield (src_cand, simphone.grapheme, simphone.translation)

class PhonSimStrings(object):
    """
    Class to return most phonetically similar strings from a list of
    SuperStrings.
    """
    def __init__(self, superstrings: list):
        self.candidates = superstrings
        self.ipas = [self.arpabet_to_ipa(elem.phoneme) for elem in superstrings]
        self.vecs = self.ipas_to_vecs(self.ipas)

    @staticmethod
    def arpabet_to_ipa(phonstring: str) -> tuple:
        """
        Method to turn a string of ARPABET chars into a
        string of IPA chars using the wordkit library.
        @param phonstring:
        @return:
        """
        #Remove token boundaries and separate into phonemes.
        arpa_string = phonstring.split()
        #Convert characters and split dipthtongs.
        ipa_tuple = tuple("".join(cmudict.cmu_to_ipa(arpa_string)))

        return ipa_tuple

    def ipas_to_vecs(self, phonstrings:list) -> list:
        """
        Method to turn a list of IPA phoneme strings into a list of
        feature vectors. The feactures are calculated with the
        wordkit library (CVTransformer)
        @param phonstrings: List with tuples of IPA phoneme strings
        @return: list with feature vectors.
        """
        self._c = CVTransformer(OneHotPhonemeExtractor, field=None)
        X = self._c.fit_transform(phonstrings)

        return [self._c.vectorize(word) for word in phonstrings]

    def most_similar(self, search, sim=0.5) -> Iterator:
        """
        Method to return the phonetically most similar strings for search.
        @param search: Class SuperString.
        @param sim: Upper bound cosine similarity.
        @return:
        """
        search_ipa = self.arpabet_to_ipa(search)
        search_vec = self._c.vectorize(search_ipa)
        distances = [distance.cosine(search_vec, elem) for elem in self.vecs]
        for i, elem in enumerate(distances):
            if elem < sim:
                yield self.candidates[i]

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

    def _tokenize(self, line: str) -> list:
        """
        Helper method to tokenize a string keepind internal apostrophes
        and hyphens.
        http://www.nltk.org/book/ch03.html#sec-tokenization
        @param line:
        @return: tokenized string
        """
        return re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", line)

    def get_lines(self):
        """
        Method to yield lines, according to parameter mode.
        @return:
        """
        for line in self._line_iter():
            if self._mode == "line":
                yield line
            elif self._mode == "token":
                yield " ".join(elem for elem in self._tokenize(line))
            elif self._mode == "no_punct":
                yield " ".join(elem for elem in self._tokenize(line) if elem not in string.punctuation)

def foo():
    """
    To timeit methods.
    @return:
    """
    phon_table = PhonPhrases("phrases.filtered3.ph.arpa.en-de")
    source_ex = "In fact every living creature is written in exactly the same set of letters and the same code".lower()
    test_ex = "Und in jedem Lebewesen ist es sozusagen aus Großbritannien genau mit den Buchstaben am selben Code".lower()
    candidates = Candidates(source_ex, test_ex)
    for match in candidates.phon_edit_translations(phon_table):
        print(match)

def foo2():
    """
    To timeit 2 method.
    @return:
    """
    phon_table = PhonPhrases("phrases.filtered3.ph.arpa.en-de")
    source_ex = "In fact every living creature is written in exactly the same set of letters and the same code".lower()
    test_ex = "Und in jedem Lebewesen ist es sozusagen aus Großbritannien genau mit den Buchstaben am selben Code".lower()
    new_candidates = Candidates(source_ex, test_ex)
    for match in new_candidates.pets(phon_table):
        print(match)

def main():

    ### 1. Get phonetic dictionary and homophones ###

    # phon_dic = PhonDict("phrases.filtered3.ph.arpa.en-de")
    # print(f"Number of types: {len(phon_dic)}")
    phon_table = PhonPhrases("phrases.filtered4.ph.arpa.en-de")
    # print("Number of phrases:", len(phon_table))

    # token_lengths = Counter(
    #     map(lambda x: x.grapheme.count(" "), phon_table.values()))
    # print(f"Number of phrases by token count:")
    # for i, count in sorted(token_lengths.items()):
    #     print(i+1, count)
    #
    # homophones = phon_dic.get_homophones()
    # homophone_phrases = phon_table.get_homophones()
    # print(f"Number of homophones: {len(homophones)}")
    # counter = 0
    # for key in homophone_phrases:
    #     if key.count(" ") >= 1:
    #         counter +=1
    # print("Number of homophone phrases:", counter)

    # Phon-Table stats (phrases by number of tokens).
    # token_lengths = Counter(map(lambda x: x.grapheme.count(" "), phon_table.values()))
    # Phon-table by number of characters of phrases (without counting whitespace).
    # char_lengths = Counter(map(lambda x: len(x.phoneme)-x.phoneme.count(" "), phon_table.values()))
    # print(char_lengths)
    # print(char_lengths[5])

    # Find phonetic neighborhoods.
    # neighborhood_ex = list(phon_table.phonetic_levenshtein("written", max_dist=1))
    # print(neighborhood_ex)
    # print(len(neighborhood_ex))

    ### 1. Get phonetic dictionary and homophones ###

    ### 2. Find phrases with smallest levenshtein distance. ###

    # print("\n")
    # print(phon_table["written"])
    # similar = list(phon_table.phonetic_levenshtein("written", max_dist=3))
    # # print([item.grapheme for item in similar])
    # print(len(similar))
    # #Additional filter with cosine similarity.
    # phonsims = PhonSimStrings(similar)
    # hits = list(phonsims.most_similar(phon_table["written"]))
    # print(len(hits))
    # for hit in hits:
    #     print(hit.grapheme)

    # print("\n")
    # print(phon_table["filmed"])
    # similar2 = list(phon_table.phonetic_levenshtein("filmed", max_dist=3))
    # print(len(similar2))
    # #Additional filter with cosine similarity.
    # phonsims2 = PhonSimStrings(similar2)
    # hits2 = list(phonsims2.most_similar(phon_table["filmed"], 0.5))
    # print(len(hits2))
    # for hit in hits2:
    #     print(hit.grapheme)

    ### 2. Find phrases with smallest levenshtein distance. ###

    ###3. Test with sentences###
    source_ex = "In fact every living creature is written in exactly the same set of letters and the same code".lower()
    test_ex = "Und in jedem Lebewesen ist es sozusagen aus Großbritannien genau mit den Buchstaben am selben Code".lower()
    target_ex = "Und zwar jedes Lebewesen verwendet die exakt gleichen Buchstaben und denselben Code".lower()
    source_ex2 = "I filmed in war zones difficult and dangerous".lower()
    test_ex2 = "Ich fühlte mich in den Kriegsgebieten schwer und gefährlich".lower()
    source_ex3 = "the captain waved me over"
    test_ex3 = "der kapitän wartete darauf"
    source_ex4 = "I was the second volunteer on the scene so there was a pretty good chance I was going to get in".lower()
    test_ex4 = "Ich war die zweite freiwillige Freiwillige am selben Ort also gab es eine sehr gute Chance das zu bekommen".lower()
    source_ex5 = "If so , that means that what we're in the middle of right now is a transition .".lower()
    test_ex5 = "Wenn ja , das bedeutet , dass das , was wir im Moment verlieren , ein Übergang ist .".lower()

    # candidates = Candidates(source_ex5, test_ex5)
    # for match in candidates.pets(phon_table):
    #     print(match)



    # print(timeit.timeit("foo2()", globals=globals(), number=5))
    # print(cProfile.run("foo2()"))

    ###3. Test with sentences###

    ###5. Test on files ###
    print("Starting evaluation...")
    with open("evaluation_pets2.txt", "w", encoding="utf-8") as out:
        src = FileReader("evaluation.en.txt", "en", mode="no_punct")
        tst = FileReader("evaluation.de.txt", "de", mode="no_punct")
        #TODO: zip is not working.
        for n, sents in enumerate(zip(src.get_lines(), tst.get_lines()), start=1):
            source, hyp = sents[0], sents[1]
            candidates = Candidates(source.lower(), hyp.lower())
            out.write(f"{str(n)}\t")
            for pet in candidates.pets(phon_table):
                out.write(f" ({pet[0]}| {pet[1]}| {pet[2]}) ")
            out.write("\n")
        #Start of new sentence pair.
        out.write("\n")

    # ###5. Test on files ###

if __name__ == "__main__":
    main()