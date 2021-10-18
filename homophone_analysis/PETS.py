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
import epitran
import Levenshtein
from nltk.metrics import edit_distance
from wordkit.features import CVTransformer, PhonemeFeatureExtractor, PredefinedFeatureExtractor, binary_features
from wordkit.corpora.corpora import cmudict
from scipy.spatial import distance
from ipapy.ipastring import IPAString
from evaluation import evaluate

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
                yield SuperString(line[0], line[1].rstrip(), line[2])

    def _phrases_by_length(self, n: int, max_len=3):
        """
        Method to yield all elements from self._store than are in range
        n+-max_dist.
        @param n: Length of searchphrase.
        @param max_len: maximum edit operations in phonetic_levenshtein
        @return:
        """
        # Add 1 due to exclusive upper limit.
        r = range(n-max_len, n+max_len+1)
        for elem in self._store.values():
            if len(elem.phoneme) in r:
                yield elem

    def phrase_to_phon(self, phrase:str, mode="IPA") -> str:
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
            if mode == "ARPA":
                g2p = G2p()
                return " ".join(g2p(phrase))
            elif mode == "IPA":
                phonemizer = epitran.Epitran("eng-Latn")
                phon_phrase = phonemizer.transliterate(phrase)
                return no_supra("".join(phon_phrase.split()))
            else:
                raise ValueError("mode must be 'ARPA' or 'IPA'")

    def phonetic_levenshtein(self, search: str, max_dist=0.75) -> Iterator:
        """
        Function to retrieve all phonetic phrases from self.vocab (phonetic
        phrase table) with a levenshtein distance <= max_dist.
        @param search: Search phrase.
        @param max_dist: Highest number of edit operations.
        @return: yields SuperString object.
        """
        search_phon = self.phrase_to_phon(search, mode="IPA")
        threshold = int(len(search_phon) * max_dist)
        # Only look at phrases of length +- max_len
        for superstring in self._phrases_by_length(len(search_phon)):
            # TODO: optimal distance?
            #Look up levenshtein distance with joined phonetized phrase.
            if PhonPhrases.weighted_distance(search_phon, superstring.phoneme) <= threshold:
                #Speedup/precision: translation to higher number of tokens is unlikely?
                #TODO: is this filter necessary? Does it make sense?
                # if superstring.grapheme.count(" ") <= search.count(" "):
                # Check if search phrase and otuput phrase are not the same.
                if search != superstring.grapheme:
                    yield superstring

    @staticmethod
    def weighted_distance(phrase1, phrase2, costs=(1,2,1)):
        """
        Levenshtein distance with modified costs using python-Levenshtein
        package base ond C++.
        @param phrase1: phrase in grapheme or phonetic representation.
        @param phrase2: phrase in grapheme or phnetic representation
        @param costs: a tuple or a list with three integers (d, i, s)
        @return:
        """
        ops = Levenshtein.editops(phrase1, phrase2)
        c = Counter(elem[0] for elem in ops)
        d = c["insert"] * costs[0]
        i = c["delete"]
        s = c["replace"] * costs[2]

        for op in ops:
            if op[0] == "insert" and op[1] == len(phrase1):
                c["insert"]-= 1
                c["insert"] += 1*costs[1]

        return d+i+s

class Candidates(object):
    """
    Class that uses a phonetic table (PhonPhrases) to search which phrases
    from src_sent have no translations in trg_sent.
    """
    def __init__(self, src_sent: str, trg_sent: str, phon_table, max_dist=0.75):
        self._src_sent = src_sent
        self._trg_sent = trg_sent
        #Set candidates and non-candidates.
        self._cands = self._filter_candidates(phon_table)
        # print(self._cands)
        self._non_cands = set(
            self._search_candidates(phon_table, mode="non_cand"))
        self._cand_phon_edits = self._yield_phon_edits(phon_table, max_dist=max_dist)

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
        #TODO: I use this function in pets too. But with different goal!
        hits = []
        # src_count = self._src_sent.count(superstring.grapheme)
        for trans in superstring.translation:
            #Check if individual tokens of translation phrase in target.
            # There might be a token in between them.
            hit = any(re.search(rf"{tok}", self._trg_sent) != None for tok in trans.split())
            if hit == True:
                hits.append(trans)
            # TODO: better hits?
            # matches = re.findall(trans, self._trg_sent)
            # if len(matches) == src_count:
            #     hits.append(trans)
        if hits == []:
            return []
        else:
            #TODO: Include all translations?
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
        keys = set(phon_table.keys())
        for src in src_grams:
            if src in keys:
                value = phon_table[src]
                #Search if translation of src_sent in trg_sent.
                trans_match = self._search_translation(value)
                if trans_match == []:
                    if mode == "cand":
                        yield value.grapheme
                else:
                    # TODO: Include all translations?
                    if mode == "non_cand":
                        yield trans_match
            else:
                yield src

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
            #Get number of tokens before phrase.
            src_pos = self._src_sent[:src_id.start()].count(" ")
            trg_pos = self._trg_sent[:trg_id.start()].count(" ")

            if abs(src_pos-trg_pos) <= n:
                return True
            else:
                return False

    def _filter_candidates(self, phon_table):
        """
        Method to filter candidates. Keep only phrases, when all of its tokens
        do not have a translation in self._trg_sent.
        @return:
        """
        candidates = list(self._search_candidates(phon_table))
        # For every candidate phrase, all tokens must not have a translation.
        new_candidates = sorted([cand for cand in candidates if \
        all(elem in candidates for elem in cand.split()) == True], key=len)
        #TODO: how can I improve this filter?
        # Keep only the ngram with all subgrams.
        # Solution by: https://stackoverflow.com/a/22221956/16607753
        filt_candidates = [j for i, j in enumerate(new_candidates) if
                           all(j not in k for k in new_candidates[i + 1:])]

        # print([cand for cand in candidates if \
        # all(elem in candidates for elem in cand.split()) == True])
        # print("\n")
        # print(filt_candidates)
        # print(f"Len unfiltered candidates: {len(candidates)}")
        # print(f"Len new candidates: {len(new_candidates)}")
        # print(f"Len filtered candidates: {len(filt_candidates)}")

        return filt_candidates

    def _yield_phon_edits(self, phon_table, max_dist=0.75) -> Iterator:
        """
        Method to yield all phrases (phon_edits) with levenshtein
        distance <= max_dist. Levenshtein distance is computed with
        phonetic representation.
        @param phon_table: Instance of PhonPhrases.
        @param max_dist: Maximal levenhstein distance to src_cand.
        @return: candidate, phonetized candidate, phon_edits
        """
        for src_cand in self._cands:
            # Phonetize src_cand.
            src_cand_phon = phon_table.phrase_to_phon(src_cand)
            # Get similar phrases with levenshtein distance on
            # phonetized phrases.
            phon_edits = phon_table.phonetic_levenshtein(src_cand, max_dist= max_dist)
            yield src_cand, src_cand_phon, phon_edits

    def _yield_simphones(self, sim=0.2, mode="IPA"):
        """
        Class to yield all phrases, that are phonetically similar to all
        candidates. Cosine similarity is calculated with phonetic phrases.
        @param sim: Threshold for cosine similarity. 0 is most similar and
        1 is most disimilar.
        @param mode: Specify IPA or ARPA for phonetic strings in
        phon_table.
        @return:
        """
        for src_cand, src_cand_phon, phon_edits in self._cand_phon_edits:
            # Additional filter with cosine similarity.
            phonsims = PhonSimStrings(list(phon_edits), mode=mode)
            for simphone in phonsims.most_similar(src_cand_phon, sim=sim):
                yield src_cand, simphone

    def pets(self, sim=0.2, mode="IPA"):
        """
        Method to yield all phonetically edited translations for a
        candidate.
        @param sim: Threshold for cosine similarity in method
        self._yield_simphones.
        @param mode: Specify IPA or ARPA for phonetic strings in
        phon_table.
        @return:
        """
        for src_cand, simphone in self._yield_simphones(sim=sim, mode=mode):
            # Check if translations in target sentence.
            trans = self._search_translation(simphone)
            if trans != []:
                # Check if translation not in gold translations.
                if all(re.search(rf"\b{trans}\b", non_cand) == None for
                       non_cand in self._non_cands) == True:
                    # Check if position of simphone and trans in
                    # sentences similar.
                    if self._soft_align(src_cand, trans) == True:
                        # TODO: only yield longest translation/phrase.
                        yield (src_cand, simphone.grapheme, simphone.translation)

class NewCandidates(object):
    """
    Class that uses a phonetic table (PhonPhrases) to search which phrases
    from src_sent have no translations in trg_sent.
    """
    def __init__(self, src_sent: str, trg_sent: str, phon_table, max_dist=0.75):
        self._src_sent = src_sent
        self._trg_sent = trg_sent
        self._trg_grams = self.grams(self._trg_sent)
        self._src_cands = self._filter_candidates(self._search_candidates(phon_table, mode="src"))
        self._trg_cands = self._trg_candidates(phon_table)
        # print(f"Length src_cands: {len(self._src_cands)}")
        print(self._src_cands)
        print(self._trg_cands)
        self._cand_phon_edits = self._yield_phon_edits(phon_table,
                                                       max_dist=max_dist)

    def grams(self, sent: str, order=3) -> list:
        """
        Helper method to return ngrams of order n up to to length.
        @param sent: Source or target sentence.
        @param order: Highest ngram order.
        @return: list with tuples:
        (phrase: str, phrase_position: tuple)
        """
        return [(" ".join(elem[1] for elem in gram), tuple(elem[0] for elem in gram))
                for gram in nltk.everygrams(list(enumerate(sent.split())), max_len=order)]

    def _search_translation(self, src_id, superstring, n=4):
        """
        Helper method to check if any of the translations of a superstring
        are in the target sentence.
        @param src_id: position of src_phrase in src_sent.
        @param superstring: instance of class Superstring.
        @param n: alignment flexibility in number of tokens.
        @return: Empty tuple if translation of trg_phrase in phrase table
        and aligned to src_id. Else, return tuple with target n-gram and
        target id.
        """
        #TODO: Replace soft alignment with proper alignments from
        # tool. Then, check if the aligned phrases correspond to phrase table.
        hits = []
        #Check if superstring has translation in target.
        for trg in self._trg_grams:
            for trans in superstring.translation:
                #Check if any token target phrases.
                #TODO: write as function? Include \b in regex?
                #TODO: Search shorter string in longer string.
                hit = any(re.search(rf"{tok}", trg[0]) != None for tok
                          in trans.split())
                #Source phrase has translation.
                if hit == True:
                    #Source and target phrase are 'aligned'.
                    #TODO: Write as separate boolean function.
                    if abs(src_id[0] - trg[1][0]) in range(0, n):
                        hits.append(trg)
        #TODO: yield only 1 target per 'alignments'?
        return hits

    def _search_candidates(self, phon_table, mode=""):
        """
        Method to check which ngrams don't have a translation in phon_table.
        @param phon_table: Instance of PhonPhrases.
        @param mode: If mode == "src" yield source candidate phrases.
        If mode == "trg" yield target non candidate phrases.
        @return: generator of source candidate strings or target non
        candidate strings.
        """
        for src in self.grams(self._src_sent):
            #Check if phrase in phrase table.
            try:
                phrase = phon_table[src[0]]
                #Pass src_id and phrase.
                trg_cand = self._search_translation(src[1], phrase)
                #Src phrase has no translation. Keep src as candidate.
                if trg_cand == []:
                    if mode == "src":
                        yield src
                #TODO better to yield trg_cands here?
                #Source phrase is translated in trg_sent.
                # Keep target phrase as non_candidate.
                else:
                    if mode == "trg":
                        yield trg_cand
            #Phrase not in phon_table.
            except KeyError:
                # Even if phrase not in phon_table, it's still possible
                # to find a PET.
                if mode == "src":
                    yield src

    def _filter_candidates(self, candidates):
        """
        Method to filter candidates. Keep only phrases, when all of its tokens
        do not have a translation in self._trg_sent.
        @param candidates: Generator with source or target candidates.
        Candidates are tuples with: (phrase, pos_sent).
        @return:
        """
        cands= [cand for cand in candidates]
        #Create list to search phrases only, without ids.
        cands_no_id = {cand[0] for cand in cands}
        # For every candidate phrase, all tokens must not have a translation.
        #All candidate phrases must be in list.
        #TODO: Can remove sorted.
        new_cands = sorted([cand for cand in cands if \
                                 all(elem in cands_no_id for elem in
                                     cand[0].split()) == True], key=lambda x: len(x[0]))
        #Create list to search phrases only, without ids.
        # new_cands_no_id = [cand[0] for cand in new_cands]

        # Keep only the ngram with all subgrams.
        # Solution by: https://stackoverflow.com/a/22221956/16607753
        # filt_candidates = [j for i, j in enumerate(new_cands) if
        #                    all(j[0] not in k for k in
        #                        new_cands_no_id[i + 1:])]

        return new_cands

    def _trg_candidates(self, phon_table):
        """
        Helper method to get target candidates.
        @return: dictionary with: key: phrase. value: [pos1, pos2, ...]
        """
        d = {}
        trg_non_cands = {trg for l in
         self._search_candidates(phon_table, mode="trg") for trg in
         l}
        trg_cands = set(self.grams(self._trg_sent)) - trg_non_cands
        for elem in trg_cands:
            try:
                d[elem[0]].append(elem[1])
            except KeyError:
                d[elem[0]] = [elem[1]]

        return d

    #TODO: Wrap the next three function in own class?
    def _yield_phon_edits(self, phon_table, max_dist=0.75) -> Iterator:
        """
        Method to yield all phrases (phon_edits) with levenshtein
        distance <= max_dist. Levenshtein distance is computed with
        phonetic representation.
        @param phon_table: Instance of PhonPhrases.
        @param max_dist: Maximal levenhstein distance to src_cand.
        @return: candidate, phonetized candidate, phon_edits
        """
        for src_cand in self._src_cands:
            # Phonetize src_cand.
            src_cand_phon = phon_table.phrase_to_phon(src_cand[0])
            # Get similar phrases with levenshtein distance on
            # phonetized phrases.
            phon_edits = phon_table.phonetic_levenshtein(src_cand[0], max_dist= max_dist)
            yield src_cand, src_cand_phon, phon_edits

    def _yield_simphones(self, sim=0.2, mode="IPA"):
        """
        Class to yield all phrases, that are phonetically similar to all
        candidates. Cosine similarity is calculated with phonetic phrases.
        @param sim: Threshold for cosine similarity. 0 is most similar and
        1 is most disimilar.
        @param mode: Specify IPA or ARPA for phonetic strings in
        phon_table.
        @return:
        """
        for src_cand, src_cand_phon, phon_edits in self._cand_phon_edits:
            # Additional filter with cosine similarity.
            phonsims = PhonSimStrings(list(phon_edits), mode=mode)
            for simphone in phonsims.most_similar(src_cand_phon, sim=sim):
                yield src_cand, simphone

    def pets(self, sim=0.2, mode="IPA"):
        """
        Method to yield all phonetically edited translations for a
        candidate.
        @param sim: Threshold for cosine similarity in method
        self._yield_simphones.
        @param mode: Specify IPA or ARPA for phonetic strings in
        phon_table.
        @return:
        """
        for src_cand, simphone in self._yield_simphones(sim=sim, mode=mode):
            # Check if translations in target sentence.
            #TODO: Could do direct check in target_candidates.
            for trans in simphone.translation:
                if trans in self._trg_cands.keys():
                    #TODO: Is it important to iterate over all positions?
                    for trg_id in self._trg_cands[trans]:
                        #Check if positions are aligned.
                        if abs(src_cand[1][0] - trg_id[0]) in range(0, 3):
                            # TODO: Should only yield 1 pet?
                            yield (src_cand[0], simphone.grapheme, simphone.translation)

class PhonSimStrings(object):
    """
    Class to return most phonetically similar strings from a list of
    SuperStrings.
    """
    def __init__(self, superstrings: list, mode="IPA"):
        self.superstrings = superstrings
        self._mode = mode
        self.ipas = [self.arpabet_to_ipa(elem.phoneme, mode=self._mode) \
                     for elem in self.superstrings]
        self.vecs = self.ipas_to_vecs(self.ipas)

    @staticmethod
    def _is_valid(phonstring: str):
        """
        Helper function to remove diacritics and suprasemental in phonstring.
        This is needed for vectorization with wordkit.
        @param phonstring:
        @return:
        """
        phonemes = []
        for phon in phonstring:
            features = IPAString(unicode_string=phon, single_char_parsing=True, ignore=True)
            rest = all([x.is_diacritic or x.is_suprasegmental for x in features])
            if rest == False:
                phonemes.append(phon)

        return tuple(phonemes)

    @staticmethod
    def arpabet_to_ipa(phonstring: str, mode="IPA") -> tuple:
        """
        Method to turn a string of ARPABET chars into a
        string of IPA chars using the wordkit library.
        @param phonstring: ARPA or IPA phone string.
        @param mode: If phonstring is ARPA return phonstring as IPA.
        Else return phonstring as IPA.
        @return:
        """
        #TODO: Split phrases with 2 or more tokens correctly.
        if mode == "ARPA":
            #Remove token boundaries and separate into phonemes.
            arpa_string = phonstring.split()
            #Convert characters and split dipthtongs.
            return tuple("".join(cmudict.cmu_to_ipa(arpa_string)))
        elif mode == "IPA":
            #TODO: do I need to join?
            phonstring = "".join(phon for phon in phonstring)
            return PhonSimStrings._is_valid(phonstring)
        else:
            raise ValueError("mode must be 'ARPA' or 'IPA'")

    def ipas_to_vecs(self, phonstrings:list) -> list:
        """
        Method to turn a list of IPA phoneme strings into a list of
        feature vectors. The feactures are calculated with the
        wordkit library (CVTransformer)
        @param phonstrings: List with tuples of IPA phoneme strings
        @return: list with feature vectors.
        """
        if len(self.superstrings) > 10:
            self._c = CVTransformer(PhonemeFeatureExtractor)
            X = self._c.fit_transform(self.ipas)
            return [self._c.vectorize(phrase) for phrase in phonstrings]

    def most_similar(self, search, sim=0.2) -> Iterator:
        """
        Method to return the phonetically most similar strings for search.
        @param search: Class SuperString.
        @param sim: Upper bound cosine similarity.
        @return:
        """
        #TODO: instead of threshold, I could yield the top n most similar phrases.
        search_ipa = self.arpabet_to_ipa(search, self._mode)
        # Hack to avoid empty grid in CVTransformer.
        if len(self.superstrings) > 10:
            search_vec = self._c.vectorize(search_ipa)
            distances = [1-distance.cosine(search_vec, elem) for elem in self.vecs]
            for i, dist in enumerate(distances):
                if dist >= sim:
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
        Helper method to tokenize a string keeping internal apostrophes
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

    # phon_dic = PhonDict("phrases.filtered3.ph.arpa.en-de")
    # print(f"Number of types: {len(phon_dic)}")
    phon_table = PhonPhrases("phrases.filtered5.ph.ipa.en-de")

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

    ### 2. Find phrases with smallest levenshtein distance. ###

    ###3. Test with sentences###

    # print(timeit.timeit("foo2()", globals=globals(), number=10))
    # print(cProfile.run("foo2()"))

    ###3. Test with sentences###

    ###5. Test on files ###
    #TODO: Rewrite as function.
    # print("Generating PETS...")
    # with open("pets.dev2.txt", "w", encoding="utf-8") as out:
    #     src = FileReader("pets.dev.en.txt", "en", mode="no_punct")
    #     tst = FileReader("pets.dev.de.txt", "de", mode="no_punct")
    #     for n, sents in enumerate(zip(src.get_lines(), tst.get_lines()), start=1):
    #         source, hyp = sents[0], sents[1]
    #         candidates = Candidates(source.lower(), hyp.lower(), phon_table, max_dist=0.75)
    #         for pet in candidates.pets(sim=0.3):
    #             out.write(f"{pet[0]}|||{pet[1]}|||{pet[2]}\t")
    #         out.write("\n")

    # ###5. Test on files ###

    # micro = evaluate("pets.dev.gold.tf.txt", "pets.dev.tf.txt")
    micro = evaluate("pets.dev.gold.txt", "pets.dev2.txt")
    print(micro)

    # print(phon_table["together"])
    # ex1 = phon_table["lately"].phoneme
    # # ex1 = phon_table.phrase_to_phon("door")
    # ex2= phon_table["light"].phoneme
    # print(ex1)
    # print(ex2)
    # ex1_edits = phon_table.phonetic_levenshtein("lately")
    # ex1_simphones = PhonSimStrings(list(ex1_edits), mode="IPA")
    # ex1_vec = ex1_simphones._c.vectorize(tuple(ex1))
    # ex2_vec = ex1_simphones._c.vectorize(tuple(ex2))
    # print(1-distance.cosine(ex1_vec, ex2_vec))

    source_ex3 = "And finally Saturday Developing your own unique personal style is a really great way to tell the world something about you without having to say a word".lower()
    test_ex3 = "Und schließlich Saturn der eigene einzigartige persönliche Stil zu entwickeln ist ein großartiger Weg der Welt etwas über Sie zu erzählen ohne ein Wort zu sagen".lower()
    source_ex4 = "My mother asked us to feel her hand".lower()
    test_ex4 = "Meine Mutter bat uns ihre Hände zu spüren".lower()
    source_ex5 = "Now the transition point happened when these communities got so close that in fact they got together and decided to write down the whole recipe for the community together on one string of DNA".lower()
    test_ex5 = "Der Wandel kam zu dem Punkt an dem diese Gruppen so zueinander kamen dass sie eigentlich gemeinsam ein vollendendes Rezept für die Gemeinschaft in der Gemeinschaft versammelten".lower()
    source_ex6 = "Well I went into the courtroom".lower()
    test_ex6 = "Ich ging in die Wohnung".lower()

    # candidates = Candidates(source_ex6, test_ex6, phon_table)
    # for pet in candidates.pets():
    #     print(pet)
    #
    # print("\n")
    # new_candidates = NewCandidates(source_ex6, test_ex6, phon_table)
    # for pet in new_candidates.pets():
    #     print(pet)

    print(timeit.timeit("foo()", globals=globals(), number=1))
    print(timeit.timeit("foo2()", globals=globals(), number=1))

if __name__ == "__main__":
    main()