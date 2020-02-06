import sys
import os
sys.path.append(os.getcwd())
import string
from collections import defaultdict

import numpy as np
import joblib

from deeppavlov import build_model
from deeppavlov.models.spelling_correction.levenshtein.levenshtein_searcher import LevenshteinSearcher, Trie, make_default_operation_costs

from utilities.read import read_dictionary, read_metaphone_file
from utilities.metaphone import transform
from utilities.hypotheses_handler import HypothesesHandler


def _preprocess_sent(sent):
    if isinstance(sent, str):
        words = sent.split()
    else:
        words = sent[:]
    normalized_words, sent_punctuation = [], defaultdict(list)
    for r, word in enumerate(words):
        normalized_word = word.lower().replace("ё", "е")
        if normalized_word not in string.punctuation:
            normalized_words.append(normalized_word)
        elif len(normalized_words) > 0:
            sent_punctuation[len(normalized_words) - 1].append(word)
    return normalized_words, words, sent_punctuation

class HypothesisSearcher:

    def __init__(self, data=None, max_distance: float=1, searcher_path=None,
                 word_replacement_data=None, metaphone_data=None, use_metaphone=False,
                 metaphone_score=1.0, duplication_cost=None):
        self.max_distance = max_distance
        self.word_replacement_data = word_replacement_data or dict()
        self.metaphone_data = metaphone_data
        self.use_metaphone = use_metaphone
        self.metaphone_score = metaphone_score
        if data is None and searcher_path is not None:
            data = joblib.load(searcher_path)
        if isinstance(data, (Trie, list, set)):
            if isinstance(data, Trie):
                print("Loading trie...")
                alphabet = data.alphabet    
            else:
                print("Reading words...")
                data = list({word.strip().lower().replace('ё', 'е') for word in data})    
                alphabet = sorted({x for word in data for x in word})
            searcher_params = self._make_searcher_params(alphabet, duplication_cost=duplication_cost)
            self.searcher = LevenshteinSearcher(alphabet, data, **searcher_params)
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

    def process_sentence(self, sent, sent_punctuation=None,
                         to_preprocess=True, to_return_preprocessed=False):
        if to_preprocess:
            processed_sent, source_sent_mapping, sent_punctuation = _preprocess_sent(sent)
        else:
            processed_sent = sent
        blocked_positions = set(sent_punctuation.keys()) if sent_punctuation is not None else set()
        hypotheses = self._generate_hypotheses(processed_sent, blocked_positions)
        if to_return_preprocessed:
            return hypotheses, processed_sent
        else:
            return hypotheses

    def _generate_hypotheses(self, words, blocked_positions):
        answer = [None] * len(words)
        for i, word in enumerate(words):
            word_candidates = {word}
            answer[i] = {(i+1, word, 0.0, "source")}
            for correction in self.word_replacement_data.get(word, []):
                answer[i].add((i+1, correction, 1.0, "list"))
                word_candidates.add(correction)
            if not self._can_be_corrected(word):
                continue
            candidates = self.searcher.search(word, d=self.max_distance)
            for candidate, cost in candidates:
                if candidate not in word_candidates:
                    answer[i].add((i+1, candidate, cost, "levenshtein"))
                    word_candidates.add(candidate)
            if self.use_metaphone and self._to_generate_metaphone_candidate(word):
                metaphone_codes = transform(word)
                for code in metaphone_codes:
                    for other in self.metaphone_data.get(code, []):
                        if other not in word_candidates:
                            answer[i].add((i+1, other, self.metaphone_score, "metaphone"))
                            word_candidates.add(other)
            if i < len(words)-1 and i not in blocked_positions:
                candidates = self.searcher.search(word + " " + words[i+1], d=self.max_distance)
                for candidate, cost in candidates:
                    answer[i].add((i + 2, candidate, cost, "join"))
        answer = [list(x) for x in answer]
        return answer


class LMSpellingCorrector:

    def __init__(self, lm_config, data, max_distance=1, word_replacement_data=None, 
                 metaphone_data=None, use_metaphone=False, metaphone_score=1.0, 
                 duplication_cost=None, max_corrections=1):
        self.searcher = HypothesisSearcher(
            data=data, max_distance=max_distance, word_replacement_data=word_replacement_data,
            metaphone_data=metaphone_data, use_metaphone=use_metaphone,
            metaphone_score=metaphone_score, duplication_cost=duplication_cost)
        self.lm = build_model(lm_config)
        self.max_corrections = max_corrections

    def batched_call(self, data, return_corrections=False, batch_size=4):
        N = len(data)
        processed_sents = [_preprocess_sent(sent) for sent in data]
        order = sorted(range(N), key=(lambda n: len(data[n][0])))
        answer, corrections, handlers = [None] * N, [None] * N, [None] * N
        for start in range(0, N, batch_size):
            curr_indexes = order[start:start+batch_size]
            batch = [processed_sents[i] for i in curr_indexes]
            curr_answer, curr_corrections, curr_handlers = self.__call__(batch, return_corrections=True, to_preprocess=False)
            for i, index in enumerate(curr_indexes):
                answer[index], corrections[index], handlers[index] = curr_answer[i], curr_corrections[i], curr_handlers[i]
        if return_corrections:
            return answer, corrections, handlers
        return answer

    def _generate_sent_candidates(self, candidates_data, words):
        words_to_search = [[word] for word in words]
        words_to_search_indexes = [{word: 0} for word in words]
        hypotheses_list = []
        for i, curr_candidates in enumerate(candidates_data):
            for j, correction, _, _ in curr_candidates:
                correction_words = correction.split()
                left, right = correction_words[0], correction_words[-1]
                left_length, right_length = len(words_to_search[i]), len(words_to_search[j-1])
                left_index = words_to_search_indexes[i].get(left, left_length)
                if left_index == left_length:
                    words_to_search_indexes[i][left] = left_length
                    words_to_search[i].append(left)
                right_index = words_to_search_indexes[j-1].get(right, right_length)
                if right_index == right_length:
                    words_to_search_indexes[j-1][right] = right_length
                    words_to_search[j-1].append(right)
                hypotheses_list.append({"begin": i, "end": j-1, "word": " ".join(words[i:j]), 
                                        "correction": correction, "left_index": left_index, 
                                        "right_index": right_index, "cost": 0.0})
        return words_to_search, hypotheses_list

    def _generate_candidates(self, data):
        data_for_searcher = [(elem[0], elem[2]) for elem in data]
        search_data = [self.searcher.process_sentence(*elem, to_preprocess=False, 
                                                      to_return_preprocessed=True)
                       for elem in data_for_searcher]
        answer = []
        for elem in search_data:
            words_to_search, curr_hypotheses = self._generate_sent_candidates(*elem)
            answer.append(HypothesesHandler(words_to_search, curr_hypotheses))
        return answer

    def __call__(self, data, return_corrections=False, to_preprocess=True):
        if to_preprocess:
            processed_data = [_preprocess_sent(sent) for sent in data]
        else:
            processed_data = data[:]
        sents = [elem[0] for elem in processed_data]
        corrected_sents = sents[:]
        are_sents_active = np.ones(shape=(len(sents),), dtype=bool)
        # covered_words = [np.zeros(shape=(len(sent),), dtype=float) for sent in sents]
        # candidates, hypotheses = self._generate_candidates(processed_data)
        hypotheses_handlers = self._generate_candidates(processed_data)
        corrections = [[] for _ in sents]
        for step in range(self.max_corrections):
            active_sent_indexes = np.where(are_sents_active)[0]
            if len(active_sent_indexes) == 0:
                break
            active_sents = [sents[i] for i in active_sent_indexes]
            active_candidates = [hypotheses_handlers[i].get_candidates() for i in active_sent_indexes]
            lm_scores = self.lm(active_sents, active_candidates)
            for i, curr_lm_scores in enumerate(lm_scores):
                index = active_sent_indexes[i]
                correction = hypotheses_handlers[index].extract_correction(curr_lm_scores)
                if correction is not None:
                    begin, end, corr = correction["begin"], correction["end"] + 1, correction["correction"]
                    corrected_sents[index] = sents[index][:begin] + corr.split() + sents[index][end+1:]
                    corrections[index].append(correction)
                else:
                    are_sents_active[index] = False
        if return_corrections:
            return corrected_sents, corrections, hypotheses_handlers
        else:
            return corrected_sents


if __name__ == "__main__":
    # words = read_dictionary("data/wordforms_clear.txt")
    metaphone_data = read_metaphone_file("data/metaphone_codes_clear.out")
    trie = joblib.load("data/wordforms_clear.trie")
    # model = HypothesisSearcher(data=None, searcher_path="data/wordforms_clear.trie",
    #                            use_metaphone=True, metaphone_data=metaphone_data)
    model = LMSpellingCorrector("configs/elmo_ru_predictor.json", data=trie,
                                use_metaphone=True, metaphone_data=metaphone_data)
    sent = "Кто-то довно хочет поселицца с права отменя"
    corrected_sents, sent_corrections = model([sent], return_corrections=True)
    with open("log.out", "w", encoding="utf8") as fout:
        for elem in sorted(sent_corrections[0].archive[0], key=lambda x: -x["gain"]):
            fout.write(str(elem) + "\n")
    
