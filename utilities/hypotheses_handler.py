import numpy as np

class HypothesesHandler:
    """
    Maintains a list of spelling correction hypotheses with actual indexes
    """

    def __init__(self, candidates, hypotheses):
        self.candidates = candidates
        self.hypotheses = hypotheses
        self.words_number = len(candidates)
        self.active_indexes = np.arange(self.words_number)
        self.archive = []

    def get_candidates(self):
        answer = [[] for _ in range(self.words_number)]
        for i, index in enumerate(self.active_indexes):
            if index >= 0:
                answer[index] = self.candidates[i]
        return answer

    def _find_best_correction(self, scores):
        best_score, best_hypo = -np.inf, None
        self.gains = []
        for hypo in self.hypotheses:
            hypo_probs = [scores[hypo["begin"]][0,hypo["left_index"]], 
                          scores[hypo["end"]][1,hypo["right_index"]]]
            default_probs = [scores[hypo["begin"]][0,0], scores[hypo["end"]][1,0]]
            gain = np.sum(np.log10(hypo_probs)) - np.sum(np.log10(default_probs)) + hypo["cost"]
            if gain > best_score:
                best_score, best_hypo = gain, hypo
            hypo["gain"] = gain
        return best_hypo, best_score

    def _filter_hypotheses(self, begin, end):
        self.archive.append(self.hypotheses[:])
        self.hypotheses = [hypo for hypo in self.hypotheses if hypo["end"] < begin or hypo["begin"] > end]
        return

    def _update_indexes(self, begin, end, words_count):
        self.active_indexes[begin:end+1] = -1
        self.active_indexes[end+1:] += (words_count - (end - begin + 1))
        return

    def extract_correction(self, scores, threshold=0.0):
        correction, gain = self._find_best_correction(scores)
        if gain < threshold:
            return None
        begin, end = correction["begin"], correction["end"]
        words_count = correction["word"].count(" ") + 1
        self._filter_hypotheses(begin, end)
        self._update_indexes(begin, end, words_count)
        return correction



