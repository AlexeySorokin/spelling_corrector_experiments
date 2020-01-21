import re
from copy import deepcopy


import sys
import os
import ujson as json

from deeppavlov.core.data.utils import download

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
PREROOT_DIR = os.path.dirname(SELF_DIR)
ROOT_DIR = os.path.dirname(PREROOT_DIR)
from nltk.tokenize import sent_tokenize, word_tokenize
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
# #####################################################
from lettercaser import LettercaserForSpellchecker
# from language_models.ELMO_inference import ELMOLM
from language_models.elmolm_from_config import ELMOLM
from dp_components.levenshtein_searcher_component import LevenshteinSearcherComponent
from language_models.utils import yo_substitutor
# due to computational error 0.0 advantage may occur as small negative number,
# usually it is a zero-hypothesis (no need for spelling correction hypothesis),
# so we need to grasp them.
ZERO_LOWER_BOUND = -1.0e-14
# when we split token into 2 tokens we hackily over-estimate likelihood from ELMO, so we multiply
# it with some parameter (logits are negative):
# TOKEN_SPLIT_LOGIT_MULTIPLICATOR = 1.5  #(SOTA 190920)
TOKEN_SPLIT_LOGIT_MULTIPLICATOR = 1.5

# error score incremented for tokens splitting
TOKEN_SPLIT_ERROR_SCORE = -2.0

# weighted distance limit for levenshtein search:
LEVENSHTEIN_MAX_DIST = 1.0

# out of vocabulary penalty
OOV_PENALTY = -1.0

# url where file with wordforms for candidates generator:
URL_TO_WORDFORMS = "http://files.deeppavlov.ai/spelling_correctors/wordforms.txt"


def clean_dialog16_sentences_from_punctuation(sentences):
    """
    Cleans sentences list from some punctuation

    We need to remove:
    # ?,() !;"
    # . if it not in the middle of word/sentence
    # - if it not in the middle of word
    # : if it not in the middle of word

    """
    output_sentences = []
    for each_s in sentences:
        if each_s.strip():
            sentence = each_s.strip().translate(str.maketrans("", "", '?,()!;"'))
            # clean from "..."
            sentence = re.sub(r'\.\.\.', '', sentence)

            if len(sentence) > 1 and sentence[-1] == ".":
                sentence = sentence[:-1]
            # clean from "."
            # tokens = sentence.split()
            tokens = word_tokenize(sentence)

            postprocessed_tokens = []
            for tok_idx, each_tok in enumerate(tokens):
                if each_tok == "-" or each_tok == "--":
                    # skip hyphens and dashes
                    continue

                if each_tok[-1] in ":-":
                    postprocessed_tokens.append(each_tok[:-1])
                else:
                    postprocessed_tokens.append(each_tok)

            reassembled_sentence = " ".join(postprocessed_tokens)

            # finally we need to remove excessive spaces?
            reassembled_sentence = re.sub(" {2,}", " ", reassembled_sentence)
            output_sentences.append(reassembled_sentence)
        else:
            output_sentences.append(each_s.strip())
    return output_sentences


class ELMO40inSpellingCorrector():
    """
    Spelling corrector based on ELMO language model
    """

    def __init__(self, language_model=None, spelling_correction_candidates_generator=None,
                 fix_treshold=10.0, max_num_fixes=5, data_path=None, mini_batch_size=None,
                 frozen_words_regex_patterns=['тинькоф+', "греф", "писец", "кв\.", "м\.", "квортира", "пирдуха"]):
        """

        :param language_model:
        :param spelling_correction_candidates_generator:
        :param fix_treshold:
        :param max_num_fixes:
        :param data_path:
        :param mini_batch_size:
        :param frozen_words_regex_patterns: list of words regexp patterns which are prohibited for correction (patterns only for tokens)
        """
        print("Init LetterCaser.")
        self._lettercaser = LettercaserForSpellchecker()
        print("Init language_model.")
        if language_model:
            self.lm = language_model
        else:

            self.lm = self._init_elmo(mini_batch_size=mini_batch_size)
        print("Init spelling_correction_candidates_generator.")
        # DATA PATH
        if data_path:
            self._data_path = data_path
        else:
            # import sys
            import os

            SELF_DIR = os.path.dirname(os.path.abspath(__file__))
            ROOT_DIR = os.path.dirname(os.path.dirname(SELF_DIR))
            DATA_PATH = ROOT_DIR + '/data'
            # = "/home/alx/Cloud/spell_corr/py_spelling_corrector/data/"
            self._data_path = DATA_PATH

        if spelling_correction_candidates_generator:
            self.sccg = spelling_correction_candidates_generator
        else:
            self.sccg = self._init_sccg()

        # max allowed number of fixes in a sentence
        self.max_num_fixes = max_num_fixes

        # minimal likelihood advantage treshold for fixing the sentence
        self.fix_treshold = fix_treshold

        if not frozen_words_regex_patterns:
            self.frozen_words_regex_patterns = []
        else:
            assert isinstance(frozen_words_regex_patterns, list)
            self.frozen_words_regex_patterns = [re.compile(pat) for pat in frozen_words_regex_patterns]
        print("Initialization Completed.")

    def _init_elmo(self, mini_batch_size=None):
        """
        Initilize default ELMO LM if no specification was provided in configuration
        :return: ELMOLM instance
        """

        # instance = ELMOLM(model_dir="~/.deeppavlov/downloads/embeddings/elmo_ru_news/")

        # TODO: azat substitute please with ELMO_inference component
        news_elmo_code = "elmo_ru_news"
        news_elmo_targz = "lm_elmo_ru_news.tar.gz"

        # news_simple_elmo_code = "elmo-lm-ready4fine-tuning-ru-news-simple"
        # news_elmo_simple_targz = "elmo-lm-ready4fine-tuning-ru-news-simple.tar.gz"

        # wiki_elmo_code = "elmo-lm-ready4fine-tuning-ru-wiki"
        # wiki_elmo_targz = "elmo-lm-ready4fine-tuning-ru-wiki.tar.gz"

        selected_model_dir = news_elmo_code
        selected_model_targz = news_elmo_targz
        # selected_model_dir = wiki_elmo_code
        # selected_model_targz = wiki_elmo_targz
        # selected_model_dir = news_simple_elmo_code
        # selected_model_targz = news_elmo_simple_targz

        if not mini_batch_size:
            mini_batch_size = 10
        with open("config/elmo_config.json", "r") as fin:
            self.elmo_config = json.load(fin)
            self.elmo_config["chainer"]["pipe"][-1]["mini_batch_size"] = mini_batch_size
            self.elmo_config["chainer"]["pipe"][-1]["model_dir"] = os.path.join("bidirectional_lms", selected_model_dir)
            self.elmo_config["metadata"]["download"][0]["url"] = "http://files.deeppavlov.ai/deeppavlov_data/" + selected_model_targz
        instance = ELMOLM(self.elmo_config)
        return instance

    def _init_sccg(self):
        """
        Initilizes Spelling Correction Candidates Generator to generate correction candidates
        :return: instance of spelling correction candidates generator
        """
        # TODO refactor with dynamic dictionary
        path_to_dictionary = self._data_path + "/wordforms.txt"

        try:
            # path_to_dictionary = DATA_PATH + "russian_words_vocab.dict"
            # TODO download if no files
            with open(path_to_dictionary, "r") as dict_file:
                # to avoid confusion: words_dict is a list of strings (which are words of
                # language's dictionary)
                self.words_dict = dict_file.read().splitlines()
        except Exception as e:
            #download it, then read it again


            download(path_to_dictionary, URL_TO_WORDFORMS, force_download=False)
            with open(path_to_dictionary, "r") as dict_file:
                # to avoid confusion: words_dict is a list of strings (which are words of
                # language's dictionary)
                self.words_dict = dict_file.read().splitlines()

        lsc = LevenshteinSearcherComponent(words=self.words_dict,
                                           max_distance=LEVENSHTEIN_MAX_DIST,
                                           oov_penalty=OOV_PENALTY
                                           )
        return lsc

    def preprocess_sentence(self, sentence):
        # lowercase
        lowercased_sentence = sentence.lower()

        # substitute ё:
        lowercased_sentence = yo_substitutor(lowercased_sentence)

        return lowercased_sentence

    def process_sentence(self, sentence):
        """
        Interface method for sentence correction.
        Given a sentence as string anlyze it, fix it and output the best hypothesis
        :param sentence: str
        :return: str, sentence with corrections
        """
        # preprocess
        preprocessed_sentence = self.preprocess_sentence(sentence)

        # analyse sentence
        analysis_dict = self.elmo_analysis_with_probable_candidates_reduction_dict_out(preprocessed_sentence)

        # TODO add support of Nto1 Hypotheses generator which updates analysis dict

        # implement the best fixes
        output_sentence = self.fixes_maker(analysis_dict, max_num_fixes=self.max_num_fixes,
                                           fix_treshold=self.fix_treshold)

        # restore capitalization:
        # output_sentence = self._lettercaser([sentence.split()], [output_sentence.split()])
        output_sentence = self._lettercaser([word_tokenize(sentence)], [word_tokenize(output_sentence)])

        return output_sentence

    # deprecated: prefer to use elmo_analysis_with_probable_candidates_reduction_dict_in_dict_out
    def elmo_analysis_with_probable_candidates_reduction_dict_out(self, sentence):
        """
        Given a sentence this method analyzes it and returns an analysis dictionary
        with hypotheses of the best substitutions (as scored lists for each token).

        This analysis accounts only fixes that contain 1-1 token conversion (without tokens
        splitting or merging).

        The analysis dictionary allows to make parametrized hypothesis selection at the next stage.

        Example of Output:
        {
            "input_sentence": "...",
            "tokenized_input_sentence": ['<S>',
                                              'обломно',
                                              'но',
                                              'не',
                                              'сдал',
                                              'горбачева',
                                              'но',
                                              'хочу',
                                              'сдать',
                                              'последний',
                                              'экзам',
                                              'на',
                                              '5',
                                              'тогда',
                                              'буит',
                                              'возможно',
                                              'хоть',
                                              'ченить',
                                              'выловить',
                                              'на',
                                              'горбачеве',
                                              '</S>']
            "word_substitutions_candidates": [
                {'tok_idx': 0,
                'top_k_candidates': [
                        {'token_str': обломно,
                        'advantage_score': 20.0
                        },
                        {'token_str': лапа,
                        'advantage_score': 21.0
                        }

                    ]
                },
                {'tok_idx': 2,
                'top_k_candidates': [
                        {'token_str': но,
                        'advantage_score': 20.1
                        },
                        {'token_str': калал,
                        'advantage_score': 21.3
                        }

                    ]
                }

            ]

        """
        result_data_dict = {
            'input_sentence': sentence
        }
        tok_wrapped = self.lm.tokenize_sentence(sentence)
        #     toks_unwrapped = tok_wrapped[1:-1]
        result_data_dict['tokenized_input_sentence'] = tok_wrapped

        elmo_data = self.lm.analyze_sentence(sentence)
        # elmo data array contains a ndarray of size: [1, len(sentence tokens), 1000000]
        return self.elmo_analysis_with_probable_candidates_reduction_dict_in_dict_out(result_data_dict, elmo_data)

    def elmo_analysis_with_probable_candidates_reduction_dict_in_dict_out(self, sentence_analysis_dict, elmo_data, filter_by_lm_lower_bound=None):
        """
        Given a sentence this method analyzes it and returns an analysis dictionary
        with hypotheses of the best substitutions (as scored lists for each token).

        This analysis accounts only fixes that contain 1-1 token conversion (without tokens
        splitting or merging).

        The analysis dictionary allows to make parametrized hypothesis selection at the next stage.
        Example of Input:
        {
            'input_sentence': 'очень классная тетка ктобы что не говорил',
            'tokenized_input_sentence': ['<S>',
                                        'очень',
                                        'классная',
                                        'тетка',
                                        'ктобы',
                                        'что',
                                        'не',
                                        'говорил',
                                        '</S>'],
        }

        Example of Output:
        {
            "input_sentence": "...",
            "tokenized_input_sentence": ['<S>',
                                              'обломно',
                                              'но',
                                              'не',
                                              'сдал',
                                              'горбачева',
                                              'но',
                                              'хочу',
                                              'сдать',
                                              'последний',
                                              'экзам',
                                              'на',
                                              '5',
                                              'тогда',
                                              'буит',
                                              'возможно',
                                              'хоть',
                                              'ченить',
                                              'выловить',
                                              'на',
                                              'горбачеве',
                                              '</S>']
            "word_substitutions_candidates": [
                {'tok_idx': 0,
                'top_k_candidates': [
                        {'token_str': обломно,
                        'advantage_score': 20.0
                        },
                        {'token_str': лапа,
                        'advantage_score': 21.0
                        }

                    ]
                },
                {'tok_idx': 2,
                'top_k_candidates': [
                        {'token_str': но,
                        'advantage_score': 20.1
                        },
                        {'token_str': калал,
                        'advantage_score': 21.3
                        }

                    ]
                }

            ]
        }
        """
        if not filter_by_lm_lower_bound:
            filter_by_lm_lower_bound = ZERO_LOWER_BOUND
        tok_wrapped = sentence_analysis_dict['tokenized_input_sentence']

        # provide case information:
        if 'tokenized_cased_input_sentence' in sentence_analysis_dict:
            tok_wrapped_cased = sentence_analysis_dict['tokenized_cased_input_sentence']
        else:
            tok_wrapped_cased = tok_wrapped
        # elmo data array contains a ndarray of size: [1, len(sentence tokens), 1000000]
        candidates_lists = self.sccg([tok_wrapped])
        # find the best substitutions in sentence from candidates sets
        candidates_list_for_sentence = candidates_lists[0]
        # base_scores = self.lm.trace_sentence_probas_in_elmo_datas_batch([elmo_data], [tok_wrapped])
        # log_probas_base = np.log10(base_scores)
        # # summated_probas_base = log_probas_base.sum(axis=1)
        # # TODO check if it is not necessary?
        # summated_probas_base = log_probas_base.sum()

        # for each candidate_list by levenshtein find top_k hypothese of susbstitutions in ELMO data
        word_substitutions_candidates = [{'tok_idx': idx, 'top_k_candidates': []} for idx, _ in
                                         enumerate(candidates_list_for_sentence)]

        #     for candi_idx, each_candidates_list in enumerate(candidates_list_for_sentence):
        #         # find scores in elmo data
        #         pass

        for tok_idx, input_token in enumerate(tok_wrapped):
            if tok_idx == 0:
                continue

            # if token in list of frozen patterns
            if self.frozen_words_regex_patterns:
                res = any(each_pat.match(tok_wrapped_cased[tok_idx]) for each_pat in self.frozen_words_regex_patterns)
                if res:
                    # matched pattern for blocking corrections, add zero hypothesis and continue
                    candidate_dict = {
                        # advantage of pure language model:
                        "lm_advantage": 0,
                        # advantage with error score:
                        "advantage": 0,

                        "token_str": tok_wrapped_cased[tok_idx],
                        # if it is zero hypothesis
                        "zero_hypothesis": True,
                        "error_score": 0,
                        "token_merges": 0,
                        "token_splits": 0
                    }
                    word_substitutions_candidates[tok_idx]['top_k_candidates'].append(
                        candidate_dict)
                    continue
            # retieve the best candidates
            # 1. retrive best from levenshtein list
            levenshtein_candidates_for_current_token = candidates_list_for_sentence[tok_idx]
            base_left_logit, base_right_logit = self.lm.retrieve_logits_of_particular_token(elmo_data,
                                                                                         tok_idx,
                                                                                         tok_wrapped[
                                                                                             tok_idx])
            base_summa = base_left_logit + base_right_logit
            # 2. retrieve absolute best for the position
            for each_candidate in levenshtein_candidates_for_current_token:

                # retrieve advantage
                #             print(each_candidate)
                candidate_str = each_candidate[1]

                # error score in logits for substitution input into corrected hypohesis:
                error_score = each_candidate[0]
                # TODO use  error_score for SCCG which can generate distant fixes

                candidate_dict = {
                    # advantage of pure language model:
                    "lm_advantage": None,
                    # advantage with error score:
                    "advantage": None,

                    "token_str": candidate_str,
                    # if it is zero hypothesis
                    "zero_hypothesis": None,
                    "error_score": None,
                    "token_merges": 0,
                    "token_splits": None
                }

                if " " in candidate_str:
                    ###########################################################################
                    # 1Token->2Tokens Case Generation
                    ###########################################################################
                    # if 1tok->2toks occurs we need to handle such hypotheses separately
                    # print("1tok->2toks hypothesis: %s" % candidate_str)
                    # TODO make a dirty hack?
                    # when we have a split of1 token into 2, then we could estimate advantage
                    # with reduced precision by estimating left word advantage by
                    # left_probas in elmo_data, and right_word by taking right proba for it.
                    # check it?
                    # mini_tokens = candidate_str.split()
                    mini_tokens = word_tokenize(candidate_str)
                    # TODO if len of splitting more than 2 then we may lose alot in precision!
                    # if len(mini_tokens) != 2:
                    #     import ipdb; ipdb.set_trace()
                    #
                    # else:
                    #     rightmost_index = len(mini_tokens) - 1
                    # assert len(mini_tokens) == 2, "Tokens count violation"

                    # to support multitoken splits
                    rightmost_index = len(mini_tokens) - 1

                    left_logit, _ = self.lm.retrieve_logits_of_particular_token(elmo_data,
                                                                                          tok_idx,
                                                                                          mini_tokens[0])
                    _, right_logit = self.lm.retrieve_logits_of_particular_token(elmo_data,
                                                                                tok_idx,
                                                                                mini_tokens[rightmost_index])
                    # provide information that token split occured:
                    candidate_dict["token_splits"] = len(mini_tokens)-1
                    # TODO modify likelihoods or error score? otherwise it overestimates
                    assert left_logit < 0, "Left logit must be negative"
                    assert right_logit < 0, "Right logit must be negative"
                    # multiply with 1.5 all logits becasue we use hacky over-estimation from ELMO
                    left_logit *= TOKEN_SPLIT_LOGIT_MULTIPLICATOR
                    right_logit *= TOKEN_SPLIT_LOGIT_MULTIPLICATOR

                    # with out error score
                    ###########################################################################
                else:
                    left_logit, right_logit = self.lm.retrieve_logits_of_particular_token(elmo_data,
                                                                                   tok_idx,
                                                                                   candidate_str)
                # with out error score
                # advantage_score = -base_summa + left_logit + right_logit
                # with error score
                lm_advantage = left_logit + right_logit - base_summa
                advantage_score = lm_advantage + error_score
                candidate_dict['lm_advantage'] = lm_advantage
                candidate_dict['advantage'] = advantage_score

                if candidate_str == tok_wrapped[tok_idx]:
                    # ZERO HYPOTHESIS case
                    candidate_dict['zero_hypothesis'] = True
                    candidate_dict['error_score'] = 0.0
                    word_substitutions_candidates[tok_idx]['top_k_candidates'].append(candidate_dict)
                    # TODO hack with abbreviations
                    # TODO refactor me!
                    # TODO refactor magic numbers
                    case = self._lettercaser.determine_lettercase(tok_wrapped_cased[tok_idx])
                    if case=='capitalize' and tok_idx>1:
                        # first token in sentence capitalization is not important
                        candidate_dict['advantage'] += 1.5
                        candidate_dict['case'] = case

                    elif case == 'upper':
                        # boost advantage of zero hypothesis?
                        # may be it is better to reduce advantage of all others?
                        candidate_dict['advantage'] += 5.0
                        candidate_dict['case'] = case

                    # abbreviation fuzzy match:
                    is_abbrev = re.match("([A-ZА-Я]\.*){2,}s?", tok_wrapped_cased[tok_idx])
                    if is_abbrev:
                        candidate_dict['advantage'] += 5.0
                        candidate_dict['is_abbrev'] = True

                    # SHORT WORDS HANDLING:
                    # if token is 1-2 letters in length
                    if len(tok_wrapped_cased[tok_idx])==1:
                        # huge adbvantage to 1 letters
                        candidate_dict['advantage'] += 4.0
                        candidate_dict['comment'] = "1letter word"

                    if len(tok_wrapped_cased[tok_idx])==2:
                        # last symbol is dot
                        candidate_dict['advantage'] += 0.5
                        candidate_dict['comment'] = "2letter word"

                    if tok_wrapped_cased[tok_idx][-1]=="." and len(tok_wrapped_cased[tok_idx])<=3:
                        # last symbol is dot
                        candidate_dict['advantage'] += 4.0
                        candidate_dict['comment'] = "short word with punctuation"

                    digits_regexp = re.compile('\d')

                    if digits_regexp.search(tok_wrapped_cased[tok_idx]):
                        # has digit
                        candidate_dict['advantage'] += 4.0
                        candidate_dict['comment'] = "has digit"

                # elif advantage_score >= ZERO_LOWER_BOUND:
                # temporarly filter by lm_advantage only:
                elif lm_advantage >= filter_by_lm_lower_bound:
                    # hypothesis satisfies the policy
                    candidate_dict['zero_hypothesis'] = False
                    candidate_dict['error_score'] = error_score

                    word_substitutions_candidates[tok_idx]['top_k_candidates'].append(candidate_dict)
                else:
                    # skip the candidate
                    pass
            word_substitutions_candidates[tok_idx]['top_k_candidates'] = sorted(word_substitutions_candidates[tok_idx]['top_k_candidates'],
                                                           key=lambda x: x['advantage'],
                                                           reverse=True)

        sentence_analysis_dict['word_substitutions_candidates'] = word_substitutions_candidates
        return sentence_analysis_dict

    #####################################################################
    @staticmethod
    def fixes_maker(analysis_data, max_num_fixes=5, fix_treshold=10.0, remove_s=True):
        """
        Function which actually makes spelling correction based on analysis of sentence with data
        about candidates.

        Outputs corrected sentence as a string.

        :param remove_s: if true then output string contains no <s> and </s> markers in output
        """

        tokens = deepcopy(analysis_data['tokenized_input_sentence'])

        best_substitutions_list = [
            {'tok_idx': tok_idx, 'best_candidate': each_tok, 'advantage': 0.0}
            for tok_idx, each_tok in enumerate(tokens)]

        # for all correction candidates find top-k fixes
        for tok_idx, each_candidates_list in enumerate(
                analysis_data['word_substitutions_candidates']):
            if len(each_candidates_list['top_k_candidates']) > 0:
                sorted_candidates_list = sorted(each_candidates_list['top_k_candidates'],
                                                key=lambda x: x['advantage'], reverse=True)
                best_candidate_dict = sorted_candidates_list[0]

                # filter by treshold
                if best_candidate_dict['advantage'] > fix_treshold:
                    best_substitutions_list[tok_idx]['best_candidate'] = best_candidate_dict[
                        'token_str']
                    best_substitutions_list[tok_idx]['advantage'] = best_candidate_dict['advantage']

        # now we have advantages
        # we should select top-k
        sorted_best_substitutions = sorted(best_substitutions_list, key=lambda x: x['advantage'],
                                           reverse=True)
        top_k_substitutions = sorted_best_substitutions[:max_num_fixes + 1]

        for each_substitution_element in top_k_substitutions:
            tokens[each_substitution_element['tok_idx']] = each_substitution_element[
                'best_candidate']

        if remove_s:
            output_str = " ".join(tokens[1:-1])
        else:
            output_str = " ".join(tokens)
        return output_str

    def analyze_sentence(self, sentence):
        return self.elmo_analysis_with_probable_candidates_reduction_dict_out(sentence)

    ##############################################################################################
    def __call__(self, input_sentences_batch):
        # TODO make optimized parallelization
        # optimization must be done at stage of ELMO calculation + analysis_dict construction
        #
        return [self.process_sentence(each_sentence) for each_sentence in input_sentences_batch]


if __name__ == '__main__':

    sc = ELMO40inSpellingCorrector()
    print(sc(['Мама мыла раду']))
    # print(sc(['Тут есть КТО НИБУДЬ', 'тут есть кто-нибудь']))
    # print(sc(['Это происходит По сейдень', 'это происходит посей день']))
    # print(sc(['По-моему', 'по моему']))