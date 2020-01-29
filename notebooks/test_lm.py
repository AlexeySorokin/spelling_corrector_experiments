import numpy as np

from deeppavlov import build_model, configs


elmo_lm = build_model("elmo_ru_news")
sents = ["Мама долго мыла грязную раму", "Пусть светит месяц , ночь темна", "В университете учится много студентов"]
sents = [x.split() for x in sents] * 1
probs = elmo_lm(sents)
elmo_lm["main"].init_states_before_all = False
new_probs = elmo_lm(sents)
for curr_probs, curr_new_probs in zip(probs, new_probs):
    for first, second in zip(curr_probs, curr_new_probs):
        print(np.max(np.abs(first.flatten() - second.flatten()), axis=-1), end=" ")
    print("")
# probs = np.reshape(probs, (2, -1, 1024))
# elmo_embedder = build_model("elmo_embedder/elmo_ru_news", download=True)
# vectors = elmo_embedder(["Мама долго мыла грязную раму".split()])[0]
# for i in range(4):
#     for j in range(2):
#         print(i, j, np.max(np.abs(probs[i,j*512:(j+1)*512]-vectors[1,j*512:(j+1)*512])))