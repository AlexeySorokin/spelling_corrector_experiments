from typing import Iterable
import sys
import string
from collections import defaultdict

import joblib

from deeppavlov.models.spelling_correction.levenshtein.levenshtein_searcher import LevenshteinSearcher, Trie, make_default_operation_costs

from utilities.read import read_dictionary

class HypothesisSearcher:

    def __init__(self, data=None, max_distance: float=1, searcher_path=None,
                 metaphone_data=None, use_metaphone=False, metaphone_score=1.0,
                 duplication_cost=None):
        self.max_distance = max_distance
        self.metaphone_data = None
        self.use_metaphone = use_metaphone
        if data is None and searcher_path is not None:
            data = joblib.load(searcher_path)
        if isinstance(data, (Trie, list, set)):
            if isinstance(data, Trie):
                print("Loading trie...")
                alphabet = data.alphabet    
            else:
                print("Reading words...")
                data = list({word.strip().lower().replace('ё', 'е') for word in data})    
                alphabet = sorted(x for word in data for x in word)
            searcher_params = self._make_searcher_params(alphabet, duplication_cost=duplication_cost)
            self.searcher = LevenshteinSearcher(data.alphabet, data, **searcher_params)
            if searcher_path is not None:
                joblib.dump(self.searcher.dictionary, searcher_path, compress=9)
        else:
            raise TypeError("`data` must be a collection of words or a Trie instance.")
        
    @property
    def alphabet(self):
        return self.searcher.alphabet

    def _make_searcher_params(self, alphabet, duplication_cost=None, max_duplication_length=5):
        answer = {"allow_spaces": True, "euristics": 2}
        operation_costs = make_default_operation_costs(alphabet, allow_spaces=True)
        if duplication_cost is not None:
            for L in range(2, max_duplication_length+1):
                for a in alphabet:
                    if a != "-":
                        operation_costs[a*L] = {a: duplication_cost}
                    operation_costs[a][a*L] = duplication_cost
        answer["operation_costs"] = operation_costs
        return answer            

    def _preprocess_sent(self, sent):
        if isinstance(sent, str):
            words = sent.split()
        else:
            words = sent[:]
        normalized_words, sent_punctuation = [], defaultdict(list)
        for r, word in enumerate(sent):
            normalized_word = word.lower().replace("ё", "е")
            if normalized_word not in string.punctuation:
                normalized_words.append(normalized_word)
            elif len(normalized_words) > 0:
                sent_punctuation[len(normalized_words)-1].append(word)
        return normalized_words, words, sent_punctuation

    def _can_be_corrected(self, word):
        if any(x not in self.alphabet for x in word):
            return False
        return True

    def _to_generate_metaphone_candidate(self, word):
        if any(x not in self.alphabet for x in word):
            return False
        if len(word) <= 3:
            return False
        return True

    def process_sentence(self, sent):
        processed_sent, source_sent_mapping, sent_punctuation = self._preprocess_sent(sent)
        hypotheses = self._generate_hypotheses(processed_sent, set(sent_punctuation.keys()))
        return hypotheses, processed_sent

    def _generate_hypotheses(self, words, blocked_positions):
        answer = [None] * len(words)
        for i, word in enumerate(words):
            answer[i] = {(i+1, word, 0.0)}
            if not self._can_be_corrected(word):
                continue
            candidates = self.searcher.search(word, d=self.max_distance)
            for candidate, cost in candidates:
                answer[i].add((i+1, candidate, cost))
            if self.use_metaphone and self._to_generate_metaphone_candidate(word):
                pass
            if i < len(words)-1 and i not in blocked_positions:
                candidates = self.searcher.search(word + " " + words[i+1], d=self.max_distance)
                for candidate, cost in candidates:
                    answer[i].add((i + 2, candidate, cost))
        answer = [list(x) for x in answer]
        return answer


if __name__ == "__main__":
    # words = read_dictionary("data/wordforms.txt")
    trie = joblib.load("dump/trie.out")
    model = HypothesisSearcher(trie)
    # model = HypothesisSearcher(searcher)
    sent = "Кто-то довно живет с права от станций"
    corrections, words = model.process_sentence(sent)
    for i, candidates in enumerate(corrections):
        for j, other, cost in candidates:
            print(i, j, "_".join(words[i:j]), other.replace(" ", "_"), "{:.1f}".format(cost))
