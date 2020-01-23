import sys
import os
from pathlib import Path

os.chdir(str(Path(__file__).parent.parent))
from utilities.read import read_dictionary
from utilities.metaphone import transform

if __name__ == "__main__":
    infile, outfile = sys.argv[1:]
    words = read_dictionary(infile)
    with open(outfile, "w", encoding="utf8") as fout:
        for r, word in enumerate(words):
            word_codes = transform(word)
            word_codes = [",".join(map(str, elem)) for elem in word_codes]
            fout.write("{}\t{}\n".format(word, " ".join(word_codes)))
            if r + 1 % 10000 == 0:
                print("{} words processed.".format(r+1))