import pandas as pd
import re

def format(filePath, modelName):

    lines = []

    words = []
    preds = []
    golds = []

    with open(filePath, 'r') as f:
        for line in f:
            if line != "\n":
                lines.append(line.strip())
    
    for i in range(0, len(lines), 3):
        word = lines[i].strip("word:  ")
        gold = lines[i+1].strip("Gold:  ")
        pred = lines[i+2].strip("Pred:  ")
        words.append(word.strip())
        preds.append(pred.strip())
        golds.append(gold.strip())


    return pd.DataFrame(list(zip(words, golds, preds)), columns=['word', 'gold', modelName + " pred"])

def analyze(gold, pred):

    earlyBoundary = []
    lateBoundary = []
    missedBoundary = []
    addedBoundary = []
    total = []


    for gold, pred in zip(gold, pred):

        gold = re.sub(r'[a-zA-Z/ ]', "", gold)
        pred = re.sub(r'[a-zA-Z/ ]', "", pred)

        early = 0
        late = 0
        missed = 0
        added = 0

        totalErrors = 0

        for i in range(len(gold)):
            if gold[i] == '1':
                if pred[i] == '1':
                    pass
                elif i > 0 and pred[i-1] == '1':
                    early += 1
                elif i < len(gold)-1 and pred[i+1] == '1':
                    late += 1
                else:
                    missed += 1

        added = sum(int(char) for char in pred) -  sum(int(char) for char in gold) if sum(int(char) for char in pred) -  sum(int(char) for char in gold) > 0 else 0

        earlyBoundary.append(early)
        lateBoundary.append(late)
        missedBoundary.append(missed)
        addedBoundary.append(added)
        total.append(early + late + missed + added)
        
    return earlyBoundary, lateBoundary, missedBoundary, addedBoundary, total
        

if __name__ == "__main__": 


    # gru = format("GRU/allErrors.txt", "gru")
    # gru['earlyBoundary'], gru['lateBoundary'], gru['missedBoundary'], gru['addedBoundary'], gru['totalErrors'] = analyze(gru['gold'], gru['gru pred'])
    # gru.to_csv("allGRUErrors.tsv", sep="\t")

    elman = format("Elman/allErrors.txt", "gru")
    elman['earlyBoundary'], elman['lateBoundary'], elman['missedBoundary'], elman['addedBoundary'], elman['totalErrors'] = analyze(elman['gold'], elman['gru pred'])
    elman.to_csv("allElmanErrors.tsv", sep="\t")

    # sameErrors = pd.read_csv("sameErrors.tsv", sep="\t", header=0)
    # differenErrors = pd.read_csv("differentErrors.tsv", sep="\t", header=0)

    # differentGRU = differenErrors[['item', 'gold', 'gru']].copy()
    # differentElman = differenErrors[['item', 'gold', 'elman']].copy()

    # sameErrors['earlyBoundary'], sameErrors['lateBoundary'], sameErrors['missedBoundary'], sameErrors['addedBoundary'], sameErrors['totalErrors'] = analyze(sameErrors['gold'], sameErrors['gru'])
    # sameErrors.to_csv("sameErrorsAnalyzed.tsv", sep="\t")

    # differentGRU['earlyBoundary'], differentGRU['lateBoundary'], differentGRU['missedBoundary'], differentGRU['addedBoundary'], differentGRU['totalErrors'] = analyze(differentGRU['gold'], differentGRU['gru'])
    # differentGRU.to_csv("differentGRUErrors.tsv", sep="\t")

    # differentElman['earlyBoundary'], differentElman['lateBoundary'], differentElman['missedBoundary'], differentElman['addedBoundary'], differentElman['totalErrors'] = analyze(differentElman['gold'], differentElman['elman'])
    # differentElman.to_csv("differentElmanErrors.tsv", sep="\t")


 


