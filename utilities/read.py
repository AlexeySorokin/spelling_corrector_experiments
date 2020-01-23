def read_dictionary(infile):
    answer = set()
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            word = line.strip()
            if any((x < "а" or x > "я") and x not in "ё-" for x in word.lower()):
                continue
            answer.add(word)
    return sorted(answer)