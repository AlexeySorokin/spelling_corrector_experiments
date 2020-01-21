from evaluate import align_sents

def read_infile(infile):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue
            answer.append(line)
    return answer

pred_file, corr_file = "data/dialog/train_input_sentences.txt", "data/dialog/train_golden_sentences.txt"
pred_sents = [x.split() for x in read_infile(pred_file)]
corr_sents = [x.split() for x in read_infile(corr_file)]
corrections = [None] * len(pred_sents)
for r, (pred_words, corr_words) in enumerate(zip(pred_sents, corr_sents)):
    alignment_indexes = align_sents(pred_words, corr_words, replace_cost=1.9, return_only_different=True)
    corrections[r] = [("_".join(pred_words[i:j]), "_".join(corr_words[k:l])) for (i, j), (k, l) in alignment_indexes]
    if r == 10:
        break
    print(" ".join(pred_words), " ".join(corr_words))
    for elem in corrections[r]:
        print(*elem)
    print("")