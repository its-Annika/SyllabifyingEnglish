import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report


def convertToLabled(cm):
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    gold = []
    pred = []

    for i in range(tn):
        gold.append(0)
        pred.append(0)
    
    for i in range(tp):
        gold.append(1)
        pred.append(1)
    
    for i in range(fp):
        gold.append(0)
        pred.append(1)
    
    for i in range(fn):
        gold.append(1)
        pred.append(0)
    
    return gold, pred
    

if __name__ == "__main__": 

    cm = np.array([[5005,1162],[750,11277]])
    gold, pred = convertToLabled(cm)

    print(classification_report(gold, pred, digits=6))
   
    #read in the data
    predictions = pd.read_csv("Elman/allOutput.txt", sep="\t", header=0, dtype=str)
    gold = pd.read_csv("processedData/test.tsv", sep="\t",  header=0, dtype=str)

    #sort by number of syllables
    #predicted, gold, syllable count
    syllableCount = gold['syllabified'].astype(str).apply(lambda x: x.count(';'))
    zipped = list(zip(predictions['numeric'].str.strip(), gold['numeric'].str.strip(), syllableCount))

    #generate classification reports for each syllable count
    bySyllable = defaultdict(list)
    for item in zipped:
        bySyllable[item[-1]].append(item)

    for key in bySyllable:
        pred = []
        gold = []
        for item in bySyllable[key]:
            pred += list(item[0])
            gold += list(item[1])
        print(f"Classification Report for {key + 1} syllables:\n")
        print(classification_report(gold, pred, digits=6))
