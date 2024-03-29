__author__ = "Cameron Palone"
__copyright__ = "Copyright 2015, Cameron Palone"
__credits__ = ["Cameron Palone"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Cameron Palone"
__email__ = "cam@cpalone.me"
__status__ = "Prototype"

import pickle
import random
import string

import nltk

def prepare_text_pos(text):
    return [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in nltk.sent_tokenize(text.lower())]


def prepare_text(text, word_list=None):
    """

    :type word_list: set
    """
    sents = []
    for sent in nltk.sent_tokenize(text):
        words = sent.split()
        processed = [word.lower() if word_list is not None and word.lower() in word_list else word for word in words]
        sents.append(processed)
    return sents

def combine_sentence(words):
    if words is None or len(words) < 1:
        return ""
    sent = words[0]
    for word in words[1:]:
        if word in string.punctuation:
            sent += word
        else:
            sent += " " + word
    return sent


class LanguageModel:
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def generate(self):
        raise NotImplementedError

    def _train_one_sentence(self, sent):
        raise NotImplementedError

    def update(self, lines, word_list):
        for line in lines:
            for sent in prepare_text(line, word_list=word_list):
                self._train_one_sentence(sent)


class BigramLM(LanguageModel):
    def __init__(self, k=0):
        self.bigrams = {}
        self.unigrams = {}
        self.n_bigram = 0
        self.n_unigram = 0

    def _insert_unigram(self, unigram):
        self.n_unigram += 1
        try:
            self.unigrams[unigram] += 1
        except KeyError:
            self.unigrams[unigram] = 1

    def _insert_bigram(self, bigram):
        for word in bigram:
            self._insert_unigram(word)
        self.n_bigram += 1
        try:
            self.bigrams[bigram] += 1
        except KeyError:
            self.bigrams[bigram] = 1

    def _train_one_sentence(self, sent):
        words = ["!BEGIN!"] + sent + ["!END!"]
        for i in range(len(words) - 1):
            self._insert_bigram((words[i], words[i + 1]))

    def _generate_word(self, prev_word):
        bigram_candidates = [bigram for bigram in self.bigrams.items() if bigram[0][0] == prev_word]
        n = sum(c for (bigram, c) in bigram_candidates)
        p = [c / float(n) for (bigram, c) in bigram_candidates]
        choice = random.random()
        s = 0
        for (i, prob) in enumerate(p):
            s += prob
            if s >= choice:
                return bigram_candidates[i][0][1]

    def generate(self):
        sentence = ["!BEGIN!"]
        while sentence[-1] != "!END!":
            sentence.append(self._generate_word(sentence[-1]))
        words = sentence[1:-1]
        return combine_sentence(words)


class TrigramBackoffLM(LanguageModel):
    def __init__(self, k=0):
        self.k = k
        self.bigrams = {}
        self.unigrams = {}
        self.trigrams = {}
        self.n_bigram = 0
        self.n_unigram = 0
        self.n_trigram = 0

    def _insert_trigram(self, trigram):
        self.n_trigram += 1
        try:
            self.trigrams[trigram] += 1
        except KeyError:
            self.trigrams[trigram] = 2

    def _insert_unigram(self, unigram):
        self.n_unigram += 1
        try:
            self.unigrams[unigram] += 1
        except KeyError:
            self.unigrams[unigram] = 2

    def _insert_bigram(self, bigram):
        self.n_bigram += 1
        try:
            self.bigrams[bigram] += 1
        except KeyError:
            self.bigrams[bigram] = 2

    def _train_one_sentence(self, sent):
        words = ["!BEGIN!", "!BEGIN!"] + sent + ["!END!"]
        for i in range(len(words) - 2):
            self._insert_trigram((words[i], words[i + 1], words[i + 2]))
        for i in range(len(words) - 1):
            self._insert_bigram((words[i], words[i + 1]))
        # Don't need beginning or end tokens
        for word in words[1:]:
            self._insert_unigram(word)

    def _p_unigram(self, cand):
        if cand not in self.unigrams:
            return 0.
        return self.unigrams[cand] / self.n_unigram

    def _p_bigram(self, prev_word, cand):
        if (prev_word, cand) not in self.bigrams:
            return None
        bi_c = self.bigrams[(prev_word, cand)]
        # TODO : Use Good-Turing estimation
        d = 1.
        uni_c = self.unigrams[cand]
        return d * float(bi_c) / uni_c

    def _p_trigram(self, prev_bigram, cand):
        if (prev_bigram[0], prev_bigram[1], cand) not in self.trigrams:
            return None
        tri_c = self.trigrams[(prev_bigram[0], prev_bigram[1], cand)]
        if tri_c < self.k:
            return None
        bi_c = self.bigrams[prev_bigram]
        d = 1.
        return d * float(tri_c) / bi_c

    def _generate_word(self, prev_bigram):
        bi_cand = {}
        tri_cand = {}
        uni_cand = {}
        for word in self.unigrams:
            tri = self._p_trigram(prev_bigram, word)
            if tri is None:
                bi = self._p_bigram(prev_bigram[1], word)
                if bi is None:
                    uni_cand[word] = self._p_unigram(word)
                else:
                    bi_cand[word] = bi
            else:
                tri_cand[word] = tri
        beta = 1 - sum(tri_cand.values())

        cands = [(word, p) for (word, p) in tri_cand.items()]
        if beta > .0001:
            alpha = sum(bi_cand.values()) / beta
            cands += [(word, p * alpha) for (word, p) in bi_cand.items()]
        choice = random.random()
        s = 0.
        for (word, p) in cands:
            s += p
            if choice < s:
                return word
        return cands[-1][0]

    def generate(self):
        sentence = ["!BEGIN!", "!BEGIN!"]
        while sentence[-1] != "!END!":
            sentence.append(self._generate_word((sentence[-2], sentence[-1])))
        return combine_sentence(sentence[2:-1])
