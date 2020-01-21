from typing import Iterable
import sys

import joblib

from deeppavlov.models.spelling_correction.levenshtein.levenshtein_searcher import LevenshteinSearcher


def read_dictionary(infile):
    answer = set()
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            word = line.strip()
            if any((x < "а" or x > "я") and x != "ё" for x in word.lower()):
                continue
            answer.add(word)
    return sorted(answer)


class HypothesisSearcher:

    def __init__(self, data=None, max_distance: float=1, searcher_path=None):
        if isinstance(data, LevenshteinSearcher):
            print("Loading searcher...")
            self.searcher = data
            self.alphabet = self.searcher.alphabet
            if searcher_path is not None:
                joblib.dump(self.searcher, searcher_path, compress=9)
        elif isinstance(data, (list, set)):
            data = list({word.strip().lower().replace('ё', 'е') for word in data})
            self.alphabet = sorted({letter for word in data for letter in word})
            self.searcher = LevenshteinSearcher(self.alphabet, data, allow_spaces=True, euristics=2)
            if searcher_path is not None:
                joblib.dump(self.searcher, searcher_path, compress=9)
        else:
            self.searcher = joblib.load(searcher_path)
            self.alphabet = self.searcher.alphabet
        self.max_distance = max_distance

    def _preprocess_sent(self, sent):
        words = sent.split()
        normalized_words = [word.lower().replace("ё", "е") for word in words]
        return normalized_words, words, dict()

    def _can_be_corrected(self, word):
        if any(x not in self.alphabet for x in word):
            return False
        return True

    def process_sentence(self, sent):
        processed_sent, source_sent_mapping, sent_punctuation = self._preprocess_sent(sent)
        hypotheses = self._generate_hypotheses(processed_sent, set(sent_punctuation.keys()))
        return hypotheses, processed_sent

    def _generate_hypotheses(self, words, blocked_positions):
        answer = [None] * len(words)
        for i, word in enumerate(words):
            answer[i] = [(i+1, word, 0.0)]
            if not self._can_be_corrected(word):
                continue
            candidates = self.searcher.search(word, d=self.max_distance)
            for candidate, cost in candidates:
                answer[i].append((i+1, candidate, cost))
            if i < len(words)-1 and i not in blocked_positions:
                candidates = self.searcher.search(word + " " + words[i+1], d=self.max_distance)
                for candidate, cost in candidates:
                    answer[i].append((i + 2, candidate, cost))
        return answer


if __name__ == "__main__":
    # words = read_dictionary("data/wordforms.txt")
    searcher = joblib.load("dump/trie.out")
    model = HypothesisSearcher(searcher)
    sent = "Я довно живу с права от станций"
    corrections, words = model.process_sentence(sent)
    for i, candidates in enumerate(corrections):
        for j, other, cost in candidates:
            print(i, j, "_".join(words[i:j]), other.replace(" ", "_"), "{:.1f}".format(cost))
