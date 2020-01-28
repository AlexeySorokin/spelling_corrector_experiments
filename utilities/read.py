from collections import defaultdict

def read_dictionary(infile):
    answer = set()
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            word = line.strip()
            if any((x < "а" or x > "я") and x not in "ё-" for x in word.lower()):
                continue
            answer.add(word)
    return sorted(answer)


def read_replacement_file(infile):
    answer = defaultdict(list)
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            try:
                left, right = line.split()
            except:
                continue
            answer[left].append(right)
    return dict(answer)


def read_metaphone_file(infile):
    answer = defaultdict(set)
    with open(infile, "r", encoding="utf8") as fin:
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if line == "":
                continue
            try:
                word, codes = line.split("\t")
            except:
                continue
            codes = [tuple(map(int, elem.split(","))) for elem in codes.split()]
            for code in codes:
                answer[code].add(word.replace("ё", "е"))
            if i % 100000 == 0:
                print(i, end=" ")
    print("")
    return answer