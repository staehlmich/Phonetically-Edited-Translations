#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

import cProfile
import itertools
import re
import string
import timeit
from collections import OrderedDict, defaultdict, Counter
from collections.abc import MutableMapping
from itertools import islice, groupby
from typing import Iterator, Iterable, Generator

import Levenshtein
import epitran
import nltk
import numpy as np
from evaluation import evaluate
from g2p_en import G2p
from ipapy.ipastring import IPAString
from scipy.spatial import distance
from wordkit.features import CVTransformer, ONCTransformer, PhonemeFeatureExtractor, \
    OneHotPhonemeExtractor, binary_features

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

class SuperString(object):
    """
    Class that saves phonetic string, grapheme string and translation of
    token or phrase.
    """
    def __init__(self, grapheme: str, phoneme: str, translation: str):
        self.grapheme = grapheme
        self.phoneme = phoneme
        self.translation = tuple(translation.split("/"))

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

    def get_homophones(self, apos=True) -> dict:
        """
        Method to group SuperString by .phoneme attribute. Separate return,
        because function is rarely used.
        @param apos: If True, keep homophones with apostraophes in
        grapheme string.
        @return: SuperString.phoneme: [SuperString.phoneme1, ..., n]
        """
        d = defaultdict(list)
        homophones = OrderedDict()
        #Reverse keys (graphemes) with values (phonemes).
        for key, value in self._store.items():
            d[value.phoneme].append(value.grapheme)
        #Add only homophones to new dictionary.
        for phoneme, grapheme in d.items():
            if apos == False:
                grapheme = [token for token in grapheme if "'" not in token]
                if len(grapheme) > 1:
                    homophones[phoneme] = grapheme
            else:
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
                yield SuperString(line[0], line[1].rstrip(), line[2])

    def _phrases_by_length(self, n: int, max_len=3):
        """
        Method to yield all elements from self._store than are in range
        n+-max_dist.
        @param n: Length of searchphrase.
        @param max_len: maximum edit operations in phonetic_levenshtein
        @return:
        """
        # Add 1 due to exclusive upper boundary.
        r = range(n-max_len, n+max_len+1)
        for elem in self._store.values():
            if len(elem.phoneme) in r:
                yield elem

    def phrase_to_phon(self, phrase:str) -> str:
        """
        Method that looks up phonetic representation in self._store.
        If phrase not in self._store, resort to DeepPhonemizer/g2p as slower method.
        @param phrase: phrase as grapheme string.
        @param mode: Choose ARPA or IPA phonetic alphabet.
        @return: ARPA or IPA representation of phrase.
        """
        try:
            return self._store[phrase].phoneme
        except KeyError:
            phonemizer = epitran.Epitran("eng-Latn")
            phon_phrase = phonemizer.transliterate(phrase)
            # Remove suprasegmental features.
            return no_supra("".join(phon_phrase.split()))

    def phonetic_levenshtein(self, search: str, max_dist=0.75) -> Iterator:
        """
        Function to retrieve all phonetic phrases from self.vocab (phonetic
        phrase table) with a levenshtein distance <= max_dist.
        @param search: Search phrase.
        @param max_dist: Highest number of edit operations.
        @return: yields SuperString object.
        """
        search_phon = self.phrase_to_phon(search)
        threshold = int(len(search_phon) * max_dist)
        # Only look at phrases of length +- max_len
        for superstring in self._phrases_by_length(len(search_phon)):
            # Look up levenshtein distance with joined phonetized phrase.
            if PhonPhrases.weighted_distance(search_phon, superstring.phoneme) <= threshold:
                # Check if search phrase and otuput phrase are not the same.
                if search != superstring.grapheme:
                    yield superstring

    @staticmethod
    def weighted_distance(phrase1, phrase2, costs=(1,2,1)):
        """
        Levenshtein distance with modified costs using python-Levenshtein
        package base ond C++.
        @param phrase1: phrase in grapheme or phonetic representation.
        @param phrase2: phrase in grapheme or phonetic representation
        @param costs: a tuple or a list with three integers (d, i, s)
        @return:
        """
        ops = Levenshtein.editops(phrase1, phrase2)
        c = Counter(elem[0] for elem in ops)

        if len(phrase1) < len(phrase2):
            for op in ops:
                # Higher costs for phrases that are longer than source phrase.
                # Does not distinguish between inserts at beginning or end.
                if op[0] == "insert":
                    if op[1] == 0:
                        c["insert"]-= 1
                        c["insert"] += 1* costs[1]
                    if op[1] == len(phrase1):
                        c["insert"] -= 1
                        c["insert"] += 1 * costs[1]

        i = c["delete"] * costs[0]
        d = c["insert"]
        s = c["replace"] * costs[2]

        return d+i+s

class Candidates(object):

    def __init__(self, src_sent: str, trg_sent: str, phon_table):
        """
        Class that uses a phonetic table (PhonPhrases) to search which phrases
        from src_sent have no translations in trg_sent.
        @param src_sent: source sentence.
        @param trg_sent: target sentence (ST output).
        @param phon_table: PhonPhrases object.
        """
        self._src_sent = src_sent
        self._trg_sent = trg_sent
        self._phon_table = phon_table
        #Set candidates and non-candidates.
        self._cands = self._filter_candidates()
        self._non_cands = set(
            self._search_candidates(phon_table, mode="non_cand"))

    def everygrams(self, sent: str, length=3) -> Iterator:
        """
        Helper method to return ngrams of order n up to to length.
        @param sent: Source or target sentence.
        @param length: Highest ngram order.
        @return:
        """
        return (" ".join(elem) for elem in nltk.everygrams(sent.split(), max_len=length))

    def _search_translation(self, superstring):
        """
        Helper method to check if any of the translations of a superstring
        are in the target sentence. Return None or the highest scored
        translation.
        @param superstring: instance of class Superstring.
        @return: [] or highest scored translation.
        """
        hits = []
        for trans in superstring.translation:
            # Check if individual tokens of translation phrase in target.
            # There might be a token in between them.
            hit = any(re.search(rf"{tok}", self._trg_sent) != None for tok in trans.split())
            if hit == True:
                hits.append(trans)

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
                     nltk.everygrams(self._src_sent.split(), max_len=3)]
        for src in src_grams:
            try:
                value = phon_table[src]
                #Search if translation of src_sent in trg_sent.
                trans_match = self._search_translation(value)
                if trans_match == []:
                    if mode == "cand":
                        yield value.grapheme
                else:
                    if mode == "non_cand":
                        yield trans_match
            # Exclude source phrases not in phon_table for higher precision.
            except KeyError:
                if mode == "cand":
                    yield src
                else:
                    pass

    def _soft_align(self, src_phrase:str, trg_phrase:str, n=3)-> bool:
        """
        Helper method to find soft alignment with regex between src_phrase and
        trg_phrase. Align by number of preceeding tokens.
        @param src_phrase:
        @param trg_phrase:
        @param n: max distance between positions.
        @return: True if preceeding number of tokens <= n.
        """
        #Get starting index of phrases.
        src_id = re.search(rf'\b{src_phrase}\b', self._src_sent)
        trg_id = re.search(rf'\b{trg_phrase}\b', self._trg_sent)

        if src_id and trg_id != None:
            # Get number of tokens before phrase.
            src_pos = self._src_sent[:src_id.start()].count(" ")
            trg_pos = self._trg_sent[:trg_id.start()].count(" ")

            if abs(src_pos-trg_pos) <= n:
                return True
            else:
                return False

    def _filter_candidates(self):
        """
        Method to filter candidates. Keep only phrases, when all of its tokens
        do not have a translation in self._trg_sent.
        @return:
        """
        candidates = list(self._search_candidates(self._phon_table))
        # For every candidate phrase, all tokens must not have a translation.
        new_candidates = sorted(set([cand for cand in candidates if \
        all(elem in candidates for elem in cand.split()) == True]), key=len)
        # Keep only the ngram with all subgrams.
        # Solution by: https://stackoverflow.com/a/22221956/16607753
        # filt_candidates = [j for i, j in enumerate(new_candidates) if
        #     all(re.match(rf"\b{j}\b", k) == None for k in new_candidates[i + 1:])]

        return new_candidates

    def _yield_simphones(self, src_cand, sim=0.2, max_dist=0.75) -> Iterator:
        """
        Class to yield all phrases, that are phonetically similar to all
        candidates. Cosine similarity is calculated with phonetic phrases.
        @param src_cand: Source candidate phrase from self._cands.
        @param sim: Threshold for cosine similarity. 0 is most dissimilar
        and 1 is most similar.
        @param max_dist: Maximal levenhstein distance to src_cand.
        @return: Yields all phrases with cosine similarity greater than sim
        with src_cand.
        """
        src_cand_phon = self._phon_table.phrase_to_phon(src_cand)
        phon_edits = list(self._phon_table.phonetic_levenshtein(src_cand, max_dist=max_dist))

        # Check if CVTransformer grid too small.
        if len(phon_edits) > 10:
            phonsims = PhonSimStrings(src_cand_phon, phon_edits)

            for simphone in phonsims.most_similar(sim=sim):
                yield simphone

    def pets(self, sim=0.2, max_dist=0.75):
        """
        Method to yield all phonetically edited translations for a
        candidate.
        @param sim: Threshold for cosine similarity in method
        self._yield_simphones.
        @param max_dist: Maximal levenhstein distance to src_cand.
        @return:
        """
        for src_cand in self._cands:
            for simphone in self._yield_simphones(src_cand, sim=sim, max_dist=max_dist):
                trans = self._search_translation(simphone)
                if trans != []:
                    # Check if translation not in gold translations.
                    if all(re.search(rf"\b{trans}\b", non_cand) == None for
                           non_cand in self._non_cands) == True:
                        # Check if position of simphone and trans in
                        # sentences are similar.
                        if self._soft_align(src_cand, trans) == True:
                            # Only yield 1 pet per source phrase.
                            yield (src_cand, simphone.grapheme, trans)
                            break

class PhonSimStrings(object):

    def __init__(self, search:str, superstrings: list):
        """
        Class to return most phonetically similar strings from a list of
        SuperStrings.
        @param search: Calculate phonetically similar phrases to this
        phonetized search term.
        @param superstrings: List of phonetically edited phrases as
        output of PhonPhrases.phonetic_levenshtein.
        """
        self.superstrings = superstrings
        self.ipas = [self._validate(elem.phoneme) for elem in self.superstrings]
        self.search_ipa = self._validate(search)
        self.ipas.append(self.search_ipa)
        self.vecs = self.ipas_to_vecs(self.ipas)

    @staticmethod
    def _validate(phonstring: str):
        """
        Helper function to remove diacritics and suprasemental in phonstring.
        This is needed for vectorization with wordkit.
        @param phonstring: Phonetic IPA string.
        @return:
        """
        phonemes = []
        for phon in phonstring:
            features = IPAString(unicode_string=phon, single_char_parsing=True, ignore=True)
            rest = all([x.is_diacritic or x.is_suprasegmental for x in features])
            if rest == False:
                phonemes.append(phon)

        return tuple(phonemes)

    def ipas_to_vecs(self, phonstrings:list) -> list:
        """
        Method to turn a list of IPA phoneme strings into a list of
        feature vectors. The feactures are calculated with the
        wordkit library (CVTransformer)
        @param phonstrings: List with tuples of IPA phoneme strings
        @return: list with feature vectors.
        """
        # Hack to avoid empty grid in CVTransformer.
        if len(self.superstrings) > 10:
            self._c = CVTransformer(PhonemeFeatureExtractor, left=True)
            X = self._c.fit_transform(self.ipas)
            return [self._c.vectorize(phrase) for phrase in phonstrings]

    def most_similar(self, sim=0.2) -> Iterator:
        """
        Method to return the phonetically most similar strings for search.
        @param sim: Upper bound cosine similarity.
        @return:
        """
        # Hack to avoid empty grid in CVTransformer.
        if len(self.superstrings) > 10:
            search_vec = self._c.vectorize(self.search_ipa)
            distances = [(i, 1-distance.cosine(search_vec, elem)) for i, elem in enumerate(self.vecs)]
            for i, dist in sorted(distances, key=lambda x: x[1], reverse=True):
                if dist != 1:
                    if dist > sim:
                        yield self.superstrings[i]

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
        Helper method to tokenize a string keeping internal apostrophes.
        http://www.nltk.org/book/ch03.html#sec-tokenization
        @param line:
        @return: tokenized string
        """
        #Regex to keep internal hyphens
        return re.findall(r"\w+(?:'\w+)*|[-.(\[]+|\S\w*", line)

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

def write_pets(sourcefile:str, targetfile:str, outfile:str, phon_table,
               params:dict):
    """
    Function that reads a source and targetfile and writes PETS
    (phonetically edited translations) to outfile.
    @param sourcefile: Path to sourcefile.
    @param targetfile: Path to targetfile.
    @param outfile: Path to outfile.
    @param phon_table: phon_table instance of class PhonPhrases
    @param params: Hyperparameter settings of class Candidates and
    PhonPhrases.
    {"max_dist": 0.75, "sim":0.3}
    @return:
    """
    print("Generating PETS...")
    src = FileReader(sourcefile, "en", mode="no_punct")
    tst = FileReader(targetfile, "de", mode="no_punct")
    with open(outfile, "w", encoding="utf-8") as out:
        for n, sents in enumerate(zip(src.get_lines(), tst.get_lines()), start=1):
            source, hyp = sents[0], sents[1]
            candidates = Candidates(source.lower(), hyp.lower(), phon_table)
            for pet in candidates.pets(sim=params["sim"], max_dist=params["max_dist"]):
                out.write(f"{pet[0]}|||{pet[1]}|||{pet[2]}\t")
            out.write("\n")

class PetsStats:
    """
    Class to analyse the output of a file containing PETs (phonetically
    edited translations).
    """
    #TODO: Phrase Counter --> which phrases caused most confusion. What is their translation.?
    #TODO: Confusion in phrases at beginning or end of phrase?
    #TODO: Homohpone confusion statistic by grapheme.
    def __init__(self, file_pets: str):
        self._file_pets = file_pets
        self._pets = list(self._read_pets())
        self.sent_pets = self._set_sents()
        self.pet_dic = self._set_pet_dic()
        self.homophones = None

    def _read_pets(self):
        """
        Method to yield a list of pets per line of self._file_pets
        @return:
        """
        with open(self._file_pets, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.rstrip().split("\t")
                if line == [""]:
                    yield []
                else:
                    yield line

    def _set_sents(self):
        """
        Method to enumerate sentences and save their pets.
        @return:
        """
        d = {n:l for n,l in enumerate(self._pets, start=1)}
        return d

    def _set_pet_dic(self):
        """
        Method to set a dictionary containing the PETs by source phrase.
        @return:
        """
        d = {}
        for sent in self._pets:
            for pet in sent:
                pet = pet.split("|||")
                try:
                    d[pet[0]].append([pet[1], pet[2]])
                except KeyError:
                    d[pet[0]] = [[pet[1], pet[2]]]
        return d

    def sent_counts(self) -> dict:
        """
        Method to get counts per sentence.
        @return:
        """
        return {i:len(pets) for i,pets in self.sent_pets.items()}

    def count_errors(self) -> int:
        """
        Method to count the sentences with at least 1 pet.
        @return:
        """
        total = 0
        counts = self.sent_counts()
        for i, count in counts.items():
            if count > 0:
                total += 1
        return total

    def total_counts(self) -> int:
        """
        Method to count all PETs found in source and target file.
        @return:
        """
        total = 0
        for i, pets in self.sent_pets.items():
            total += len(pets)
        return total

    def top_errors(self, n=10):
        """
        Method to print the phrases, that were most misrecognized.
        Idea by: https://stackoverflow.com/a/16868476/16607753
        @param n: Number of top phrases to print.
        @return:
        """
        for k in sorted(self.pet_dic, key=lambda k: len(self.pet_dic[k]), reverse=True)[:n]:
            print(k)

    def counts_by_order(self):
        """
        Method to count total number of PETs by token count.
        @return:
        """
        d = {1:0, 2:0, 3:0}
        for key in self.pet_dic:
            tokens = key.count(" ")+1
            d[tokens] +=1
        return d

    def get_homophones(self, phon_table):
        """
        Method to list pets by source homophone phrase.
        @param phon_table:
        @return: ASR confusions by source homophone phrase.
        """
        self.homophones = {}
        homophones = {h for value in
                      phon_table.get_homophones().values() for h in
                      value}
        for pets in self.sent_pets.values():
            for pet in pets:
                pet = pet.split("|||")
                if pet[0] in homophones:
                    try:
                        self.homophones[pet[0]].append(pet[1])
                    except KeyError:
                        self.homophones[pet[0]] = [pet[1]]
        return self.homophones

    def homophone_types(self):
        """
        Method to count the number of errors by homophone type.
        @param phon_table:
        @return:
        """
        return len(self.homophones.values())

    def homophone_counts(self):
        """
        Method to get the total counts of homophone errors.
        @param phon_table:
        @return:
        """
        count = 0
        for key, value in self.homophones.items():
            count += len(value)
        return count

    def homophones_by_order(self):
        """
        Method to count total number of homophone PETs by token count.
        @return:
        """
        d = {1:0, 2:0, 3:0}
        for key in self.homophones:
            tokens = key.count(" ")+1
            d[tokens] +=1
        return d

def foo():
    """
    To timeit methods.
    @return:
    """
    phon_table = PhonPhrases("phrases.filtered5.ph.ipa.en-de")
    source_ex3 = "And finally Saturday Developing your own unique personal style is a really great way to tell the world something about you without having to say a word".lower()
    test_ex3 = "Und schließlich Saturn der eigene einzigartige persönliche Stil zu entwickeln ist ein großartiger Weg der Welt etwas über Sie zu erzählen ohne ein Wort zu sagen".lower()
    new_candidates = Candidates(source_ex3, test_ex3, phon_table)
    for pet in new_candidates.pets():
        print(pet)

def foo2():
    """
    To timeit 2 method.
    @return:
    """
    phon_table = PhonPhrases("phrases.filtered5.ph.ipa.en-de")
    source_ex3 = "And finally Saturday Developing your own unique personal style is a really great way to tell the world something about you without having to say a word".lower()
    test_ex3 = "Und schließlich Saturn der eigene einzigartige persönliche Stil zu entwickeln ist ein großartiger Weg der Welt etwas über Sie zu erzählen ohne ein Wort zu sagen".lower()
    new_candidates = NewCandidates(source_ex3, test_ex3, phon_table)
    for pet in new_candidates.pets():
        print(pet)

def main():

    ### 1. Get phonetic dictionary and homophones ###

    phon_dic = PhonDict("phrase_table_filtered.en-de")
    # print(f"Number of types: {len(phon_dic)}")
    phon_table = PhonPhrases("phrase_table_filtered.ipa.en-de")

    # print("Number of phrases:", len(phon_table))

    # token_lengths = Counter(
    #     map(lambda x: x.grapheme.count(" "), phon_table.values()))
    # print(f"Number of phrases by token count:")
    # for i, count in sorted(token_lengths.items()):
    #     print(i+1, count)
    #
    # homophones = phon_dic.get_homophones(apos=False)
    # homophone_phrases = phon_table.get_homophones(apos=False)

    # print(f"Number of homophones: {len(homophones)}")
    # counter = 0
    # for key, value in homophone_phrases.items():
    #     print(value)
    #     if key.count(" ") >= 1:
    #         counter +=1
    # print(f"Number of homophone phrases: {len(homophone_phrases)}")

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
    # similar = list(phon_table.phonetic_levenshtein("thursday", max_dist=0.75))
    # for item in similar:
    #     print(item.grapheme, item.phoneme)
    # similar.append(phon_table["thursday"])
    # print([item.grapheme for item in similar])
    # print(len(similar))
    #Additional filter with cosine similarity.
    # phonsims = PhonSimStrings(phon_table["thursday"].phoneme, similar)
    # print(phonsims._validate(phon_table["is written"].phoneme))
    # print(phonsims.arpabet_to_ipa(phon_table["view"].phoneme))
    # hits = list(phonsims.most_similar(sim=0.2))
    # for hit in hits:
    #     print(hit)
    # print(len(hits))


    ### 2. Find phrases with smallest levenshtein distance. ###

    ###3. Test with sentences###

    source_ex2 = "And if we show that to people then we can also promote some behavioral change".lower()
    test_ex2 = "Wir zeigen es den Leuten dann können wir auch unser Verhalten fördern".lower()
    source_ex3 = "And finally Saturday Developing your own unique personal style is a really great way to tell the world something about you without having to say a word".lower()
    test_ex3 = "Und schließlich Saturn der eigene einzigartige persönliche Stil zu entwickeln ist ein großartiger Weg der Welt etwas über Sie zu erzählen ohne ein Wort zu sagen".lower()
    source_ex4 = "My mother asked us to feel her hand".lower()
    test_ex4 = "Meine Mutter bat uns ihre Hände zu spüren".lower()
    source_ex5 = "Now the transition point happened when these communities got so close that in fact they got together and decided to write down the whole recipe for the community together on one string of DNA".lower()
    test_ex5 = "Der Wandel kam zu dem Punkt an dem diese Gruppen so zueinander kamen dass sie eigentlich gemeinsam ein vollendendes Rezept für die Gemeinschaft in der Gemeinschaft versammelten".lower()
    source_ex6 = "Well I went into the courtroom".lower()
    test_ex6 = "Ich ging in die Wohnung".lower()
    source_ex7 = "Thursday Confidence is key".lower()
    test_ex7 = "Dritter Tag Kompetenz ist der Schlüssel".lower()

    # candidates = Candidates(source_ex7, test_ex7, phon_table)
    # for pet in candidates.pets(sim=0.2, max_dist=0.8):
    #     print(pet)

    # print("\n")
    # new_candidates = NewCandidates(source_ex2, test_ex2, phon_table)
    # for pet in new_candidates.pets():
    #     print(pet)
    # # print(phon_table["behavioral"])
    # print(timeit.timeit("foo()", globals=globals(), number=2))
    # print(timeit.timeit("foo2()", globals=globals(), number=1))

    ###3. Test with sentences###

    ###5. Test and evaluate on dev files ###

    best_params = {"max_dist": 0.6, "sim": 0.7}
    # write_pets("pets.dev.en.txt", "pets.dev.de.detok.txt", "pets.dev6.txt", phon_table, best_params)
    # write_pets("pets.test.en.txt", "pets.test.de.txt", "pets.test.txt", phon_table, best_params)

    # micro = evaluate("pets.dev.gold.tf.txt", "pets.dev.tf.txt")
    # micro_dev = evaluate("pets.dev.gold.txt", "pets.dev6.txt")
    # micro_dev = evaluate("pets.dev.gold.txt", "pets.dev7.txt")

    # micro_test = evaluate("pets.test2.gold.txt", "pets.test.txt")
    # print(micro_dev)
    # print(micro_test)

    ###5. Test and evaluate on dev files ###

    ###6. Run PETs on output of ST systems.###

    test_en = "/home/user/staehli/master_thesis/data/MuST-C/en-de/data/tst-COMMON/txt/tst-COMMON.en"
    # cascade = "/home/user/staehli/master_thesis/pets/output_models/cascade.detok.txt"
    # write_pets(test_en, cascade, "pets.cascade3.txt", phon_table, best_params)

    # afs = "/home/user/staehli/master_thesis/pets/output_models/afs.detok.txt"
    # write_pets(test_en, afs, "pets.afs3.txt", phon_table, best_params)

    # context = "/home/user/staehli/master_thesis/pets/output_models/docmodel-imed.detok.txt"
    # write_pets(test_en, context, "pets.context3.txt", phon_table,best_params)

    test_en_v2 = "/home/user/staehli/en-de/data/tst-COMMON/txt/tst-COMMON.en"
    # afs2_v1 = "/home/user/staehli/master_thesis/output/afs_tf_mustc_v2/trans.detok.v1.txt"
    # write_pets(test_en, afs2_v1, "pets.afs2_v1.txt", phon_table, best_params)

    # afs2_v2 = "/home/user/staehli/master_thesis/output/afs_tf_mustc_v2/trans.detok.v2.txt"
    # write_pets(test_en_v2, afs2_v2, "pets.afs2_v2.txt", phon_table, best_params)

    # afs_v2 = "/home/user/staehli/master_thesis/output/afs_tf/trans.detok.v2.txt"
    # write_pets(test_en_v2, afs_v2, "pets.afs_v2.txt", phon_table, best_params)


    ###6. Run PETs on output of ST systems.###

    ###7. Analyse output of PETs files.###
    # sent_num = 6
    # cascade_pets = PetsStats("pets.cascade2.txt")
    # print(cascade_pets.sent_pets[sent_num])
    # print(f"Sentences with PETs cascade: {cascade_pets.count_errors()}")
    # print(f"Total PETs cascade: {cascade_pets.total_counts()}")
    # print(f"Unique source phrases cascade: {len(cascade_pets.pet_dic)}")
    # print(f"Counts by order cascade: {cascade_pets.counts_by_order()}")
    # cascade_pets.top_errors(n=10)
    # cascade_homophones = cascade_pets.get_homophones(phon_table)
    # print(cascade_homophones)
    # print(f"Number of homophone type errors cascade: {cascade_pets.homophone_types()}")
    # print(f"Total number of homophone errors cascade: {cascade_pets.homophone_counts()}")
    # print(f"Homophones by order cascade: {cascade_pets.homophones_by_order()}")
    # print("\n")
    # afs_pets = PetsStats("pets.afs.txt")
    # print(afs_pets.sent_pets[sent_num])
    # print(f"Sentences with PETs AFS: {afs_pets.count_errors()}")
    # print(f"Total PETs AFS: {afs_pets.total_counts()}")
    # print(f"Unique source phrases AFS: {len(afs_pets.pet_dic)}")
    # print(f"Counts by order AFS: {afs_pets.counts_by_order()}")
    # afs_pets.top_errors(n=10)
    # afs_homophones = afs_pets.get_homophones(phon_table)
    # print(afs_homophones)
    # print(f"Number of homophone type errors AFS: {afs_pets.homophone_types()}")
    # print(f"Total number of homophone types errors AFS: {afs_pets.homophone_counts()}")
    # print(f"Homophones by order AFS: {afs_pets.homophones_by_order()}")
    # print("\n")
    # context_pets = PetsStats("pets.context.txt")
    # print(context_pets.sent_pets[sent_num])
    # print(f"Sentences with PETs Context: {context_pets.count_errors()}")
    # print(f"Total PETs Context: {context_pets.total_counts()}")
    # print(f"Unique source phrases Context: {len(context_pets.pet_dic)}")
    # print(f"Counts by order Context: {context_pets.counts_by_order()}")
    # context_pets.top_errors(n=10)
    # context_homophones = context_pets.get_homophones(phon_table)
    # print(context_homophones)
    # print(f"Number of homophone type errors Context: {context_pets.homophone_types()}")
    # print(f"Total number of homophone errors Context: {context_pets.homophone_counts()}")
    # print(f"Homophones by order Context: {context_pets.homophones_by_order()}")

     # print("\n")
    #Sentences were all systems make mistakes.
    # counter = 0
    # for n in cascade_pets.sent_pets:
    #     casc = cascade_pets.sent_pets[n]
    #     afs = afs_pets.sent_pets[n]
    #     ctxt = context_pets.sent_pets[n]
    #
    #     if casc and afs and ctxt != []:
    #         print(n)
    #         print(casc)
    #         print(afs)
    #         print(ctxt)
    #         print("\n")
    #         counter += 1
    # print(counter)

    ##7. Analyse output of PETs files.###

if __name__ == "__main__":
    main()