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
from wordkit.features import CVTransformer, PhonemeFeatureExtractor
from wordkit.corpora.corpora import cmudict
from scipy.spatial import distance
from ipapy.ipastring import IPAString
from evaluation import evaluate

class SuperString(object):
    """
    Class that saves phonetic string, grapheme string and translation of
    token or phrase.
    """
    def __init__(self, grapheme: str, phoneme: str, translation: str):
        self.grapheme = grapheme
        self.phoneme = phoneme
        #TODO: Phoneme string without whitespaces.
        self.joined_phoneme = self.join_phon()
        #TODO: translations in tuple.
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
            if len(elem.joined_phoneme) in r:
                yield elem

    def phrase_to_phon(self, phrase:str, mode="ARPA") -> str:
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
            elif mod == "IPA":
                phonemizer = epitran.Epitran("eng-Latn")
                return phonemizer.transliterate(phrase)
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
        joined_search = "".join(elem for elem in search_phon.split())
        # Only look at phrases of length +- max_len
        for superstring in self._phrases_by_length(len(joined_search)):
            # TODO: optimal distance?
            threshold = int(len(joined_search)*max_dist)
            #Look up levenshtein distance with joined phonetized phrase.
            if PhonPhrases.weighted_distance(joined_search, superstring.joined_phoneme) <= threshold:
                # if self.longest_token(search,superstring.grapheme) == False:
                #Speedup/precision: translation to higher number of tokens is unlikely?
                #TODO: is this filter necessary? Does it make sense?
                # if superstring.grapheme.count(" ") <= search.count(" "):
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
    #TODO: parameter setting in class call?
    def __init__(self, src_sent: str, trg_sent: str, phon_table, max_dist=0.75):
        #TODO: time if phon_table as parameter?
        self._src_sent = src_sent
        self._trg_sent = trg_sent
        #Set candidates and non-candidates.
        self._cands = self._filter_candidates(phon_table)
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
            #Check if individual tokens of translation phrase in target.
            #There might be a token in between them.
            hit = any(re.search(rf"{tok}", trg_sent) != None for tok in trans.split())
            if hit == True:
                hits.append(trans)
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
                #Search source phrase in source sentence. Not needed. Done in src_grams.
                # if re.search(rf"\b{src}\b", self._src_sent) != None:
                #Search if translation of src_sent in trg_sent.
                trans_match = self._search_translation(value, self._trg_sent)
                if trans_match == []:
                    if mode == "cand":
                        yield value.grapheme
                else:
                    if mode == "non_cand":
                        yield trans_match

    def _soft_align(self, src_phrase:str, trg_phrase:str, src_sent:str, trg_sent:str, n=4)-> bool:
        """
        Helper method to find soft alignment with regex between src_phrase and
        trg_phrase. Align by number of preceeding tokens.
        @param src_phrase:
        @param trg_phrase:
        @param src_sent:
        @param trg_sent:
        @param n: max distance between positions.
        @return: True if preceeding number of tokens <= n.
        """
        #Get starting index of phrases.
        src_id = re.search(rf'\b{src_phrase}\b', src_sent)
        trg_id = re.search(rf'\b{trg_phrase}\b', trg_sent)
        if src_id and trg_id != None:
            #Get number of tokens before phrase.
            src_pos = src_sent[:src_id.start()].count(" ")
            trg_pos = trg_sent[:trg_id.start()].count(" ")

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
        new_candidates = [cand for cand in candidates if all(elem in candidates for elem in cand.split()) == True]
        # Keep only the longest strings.
        # Solution by: https://stackoverflow.com/a/22221956/16607753
        new_candidates = [j for i, j in enumerate(new_candidates) if
         all(j not in k for k in new_candidates[i + 1:])]

        return new_candidates

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

    def _yield_simphones(self, sim=0.6, mode="IPA"):
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

    def pets(self, sim=0.6, mode="IPA"):
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
            trans = self._search_translation(simphone, self._trg_sent)
            if trans != []:
                # Check if translation not in gold translations.
                if all(re.search(rf"\b{trans}\b", non_cand) == None for
                       non_cand in self._non_cands) == True:
                    # Check if position of simphone and trans in
                    # sentences similar.
                    if self._soft_align(src_cand, trans, self._src_sent,
                                        self._trg_sent) == True:
                        # TODO: only yield longest translation/phrase.
                        yield (src_cand, simphone.grapheme, simphone.translation)

class PhonSimStrings(object):
    """
    Class to return most phonetically similar strings from a list of
    SuperStrings.
    """
    def __init__(self, superstrings: list, mode="ARPA"):
        self.superstrings = superstrings
        self._mode = mode
        self.ipas = [self.arpabet_to_ipa(elem.phoneme, mode=self._mode) \
                     for elem in superstrings]
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
    def arpabet_to_ipa(phonstring: str, mode="ARPA") -> tuple:
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

            return [self._c.vectorize(word) for word in phonstrings]

    def most_similar(self, search, sim=0.6) -> Iterator:
        """
        Method to return the phonetically most similar strings for search.
        @param search: Class SuperString.
        @param sim: Upper bound cosine similarity.
        @return:
        """
        search_ipa = self.arpabet_to_ipa(search, self._mode)
        # Hack to avoid empty grid in CVTransformer.
        if len(self.superstrings) > 10:
            search_vec = self._c.vectorize(search_ipa)
            distances = [distance.cosine(search_vec, elem) for elem in self.vecs]
            for i, dist in enumerate(distances):
                if dist <= sim:
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
    phon_table = PhonPhrases("phrases.filtered4.ph.arpa.en-de")
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
    phon_table = PhonPhrases("phrases.filtered4.ph.ipa.en-de")
    s1 = phon_table["set of"].phoneme
    s2 = phon_table["set"].phoneme

    phon_table.weighted_distance(s1, s2)

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
    source_ex2 = "I filmed in war zones difficult and dangerous".lower()
    test_ex2 = "Ich fühlte mich in den Kriegsgebieten schwer und gefährlich".lower()
    source_ex3 = "the captain waved me over"
    test_ex3 = "der kapitän wartete darauf"
    source_ex4 = "I was the second volunteer on the scene so there was a pretty good chance I was going to get in".lower()
    test_ex4 = "Ich war die zweite freiwillige Freiwillige am selben Ort also gab es eine sehr gute Chance das zu bekommen".lower()
    source_ex5 = "In my warped little industry , that's my brand .".lower()
    test_ex5 = "Das ist mein Gehirn .".lower()

    # candidates = Candidates(source_ex, test_ex, phon_table)
    # for match in candidates.pets():
    #     print(match)

    # print("\n")
    # print(phon_table["filmed"])
    # print(phon_table["felt"])
    # print(Levenshtein.distance('fɪlmd','fɛlt'))
    # p1 = phon_table["mary"].phoneme
    # p2 = phon_table["amerika"].phoneme
    # print(p1, p2)
    # print(phon_table.phonetic_levenshtein(p1, p2))
    # print(phon_table["that's"])
    s1 = phon_table["filmed"].phoneme
    s2 = phon_table["felt"].phoneme
    s3 = phon_table["using"].phoneme
    # s1_ipa = PhonSimStrings.arpabet_to_ipa(s1)
    # s2_ipa = PhonSimStrings.arpabet_to_ipa(s2)
    # print(s1, s2)

    # print(timeit.timeit("foo2()", globals=globals(), number=10))
    # print(cProfile.run("foo2()"))

    ###3. Test with sentences###

    ###5. Test on files ###
    #TODO: Rewrite as function.
    print("Generating PETS...")
    with open("evaluation_pets2.txt", "w", encoding="utf-8") as out:
        src = FileReader("evaluation2.en.txt", "en", mode="no_punct")
        tst = FileReader("evaluation2.de.txt", "de", mode="no_punct")
        # TODO: zip is not working.
        for n, sents in enumerate(zip(src.get_lines(), tst.get_lines()), start=1):
            source, hyp = sents[0], sents[1]
            candidates = Candidates(source.lower(), hyp.lower(), phon_table, max_dist=0.75)
            for pet in candidates.pets(sim=0.3):
                out.write(f"{pet[0]}|||{pet[1]}|||{pet[2]}\t")
            out.write("\n")

    # ###5. Test on files ###
    # print(phon_table["beautician"].phoneme)
    # print(phon_table["petition"].phoneme)


    micro, macro = evaluate("gold_pets.txt", "evaluation_pets2.txt")
    print(micro)
    # print(phon_table["my interest"])
    # print(phon_table["my first"])



if __name__ == "__main__":
    main()