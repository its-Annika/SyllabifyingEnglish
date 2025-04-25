import re
import pandas as pd
from collections import defaultdict


def summarize(column, filename):
    
    words = column.to_list()
    typeDict = defaultdict(lambda:0)
    syllableCount = defaultdict(lambda:0)
    totalWords = 0

    for word in words:
        totalWords += 1
        syllableCount[str(word).count(";") + 1] += 1
        for syllable in str(word).split(";"):
            typeDict[syllable] += 1
    
    #the most frequent syllables
    mostFreq = sorted(typeDict.items(), key=lambda item: item[1], reverse=True)

    #number of types
    numTypes = len(typeDict.keys())

    #number of tokens
    numTokens = sum(item for item in typeDict.values())

    #average number of syllables per word
    avgSyllbles = numTokens/totalWords

    
    with open(filename, "w") as f:
        f.write(f"total number of words: {totalWords}\n")
        f.write(f"average number of syllables per word: {avgSyllbles}\n")
        f.write(f"\n\n")
        f.write(f"number of syllable types: {numTypes}\n")
        f.write(f"number of syllable tokens: {numTokens}\n")
        f.write(f"\n\n")
        f.write(f"Top 200 most frequent types:\n")
        for i in range(200):
            f.write(mostFreq[i][0] + ", " + str(mostFreq[i][1]) + "\n")



if __name__ == "__main__":

    train = pd.read_csv("train.tsv", sep="\t", header=0)
    dev = pd.read_csv("dev.tsv", sep="\t", header=0)
    test = pd.read_csv("test.tsv", sep="\t", header=0)
    all = pd.read_csv("pairs.tsv", sep="\t", header=0)

    summarize(train['syllabified'], "trainInfo.txt")
    summarize(test['syllabified'], "testInfo.txt")
    summarize(dev['syllabified'], "devInfo.txt")
    summarize(all['syllabified'], "allInfo.txt")
