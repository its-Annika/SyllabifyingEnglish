import pandas as pd

def format(filePath):
    
    errorDict = {}
    lines  = []

    with open(filePath, 'r') as f:
        for line in f:
            if line != "\n":
                lines.append(line.strip())
    
    for i in range(0, len(lines), 3):
        word = lines[i].strip("word:  ")
        gold = lines[i+1].strip("Gold:  ")
        pred = lines[i+2].strip("Pred:  ")
        errorDict[word] = (gold, pred)
    
    return errorDict

def overlap(elmanDict, gruDict):

    common = list(set(elmanDict.keys()) & set(gruDict.keys()))

    print(f"The elman model and gru model made errors on {len(common)} of the same words")

    sameErrordf = pd.DataFrame(columns=["item", 'gold', 'gru', 'elman'])
    differentErrordf = pd.DataFrame(columns=["item", 'gold', 'gru', 'elman'])
    
    index = 0
    for item in common:
        gru = gruDict[item][1]
        elman = elmanDict[item][1]
        gold = gruDict[item][0]
        same = elman == gru

        if same:
            sameErrordf.loc[index] = [item, gold, gru, elman]
            index += 1
        else:
            differentErrordf.loc[index] = [item, gold, gru, elman]
            index += 1

    print(f"In {sameErrordf.shape[0]} cases, the models made identical errors.")
    print(f"In {differentErrordf.shape[0]} cases, the models made different errors.")

    sameErrordf.to_csv("analysis/sameErrors.tsv", sep = "\t", index = True)
    differentErrordf.to_csv("analysis/differentErrors.tsv", sep = "\t", index = True)


if __name__ == "__main__": 

    elmanErrorFile = "Elman/allErrors.txt"
    gruErrorFile = "GRU/allErrors.txt"

    #dict keys - word
    #dict values - (gold, pred)
    elmanErrorDict = format(elmanErrorFile)
    gruErrorDict = format(gruErrorFile)

    print(f"The Elman model made {len(elmanErrorDict.keys())} errors.")
    print(f"The GRU model made {len(gruErrorDict.keys())} errors.")

    overlap(elmanErrorDict, gruErrorDict)


