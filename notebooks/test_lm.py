from time import time

import sys
import numpy as np

from deeppavlov import build_model, configs
from deeppavlov.dataset_readers.morphotagging_dataset_reader import read_infile

sents, _ = map(list, zip(*(read_infile("/home/Data/UD2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu"))))
test_sents = [x[:10] for x in sents[:32]]
# print([len(sent) for sent in test_sents])
elmo_lm = build_model("config/elmo_ru_news.json")
elmo_lm["main"].output_layer = "softmax"
elmo_lm["main"].single_pass = False
# sents = ["Мама долго мыла грязную раму", "Пусть светит месяц , ночь темна", "В университете учится много студентов"]
# sents = [x.split() for x in sents] * 7
t1 = time()
probs = elmo_lm(test_sents)
t2 = time()
print("{:.2f}".format(t2 - t1))
# for elem in probs:
#     print(elem.shape)
elmo_lm["main"].init_states_before_all = False
elmo_lm["main"].single_pass = True
t3 = time()
new_probs = elmo_lm(test_sents)
t4 = time()
print("{:.2f}".format(t4 - t3))
max_diff = 0.0
for curr_probs, curr_new_probs in zip(probs, new_probs):
    for first, second in zip(curr_probs, curr_new_probs):
        first, second = first.flatten(), second.flatten()
        diff = np.abs(first - second)
        size = np.maximum(np.maximum(np.abs(first), np.abs(second)), np.array([1e-12] * len(first), dtype="float"))
        max_diff = max(np.max(diff / size), max_diff)
    #     print(np.max(diff / size, axis=-1), end=" ")
    # print("")
print("{:.5f}".format(max_diff))
# probs = np.reshape(probs, (2, -1, 1024))
# elmo_embedder = build_model("elmo_embedder/elmo_ru_news", download=True)
# vectors = elmo_embedder(["Мама долго мыла грязную раму".split()])[0]
# for i in range(4):
#     for j in range(2):
#         print(i, j, np.max(np.abs(probs[i,j*512:(j+1)*512]-vectors[1,j*512:(j+1)*512])))