import sys
import re
import pandas as pd
import numpy as np

data = pd.read_csv(sys.argv[1], header=0, delimiter='\t', dtype=str)
data['syllabified'] = np.zeros(data.shape[0],)

for index, row in data.iterrows():
    letters = str((row['notSyllabified']))
    numbers = str((row['numeric']))

    reconstructed = ''
    for i in range(len(numbers)):
        if numbers[i] == "0":
            reconstructed += letters[i]
        else:
            reconstructed += ";"+letters[i]
    
    data.at[index,'syllabified'] = reconstructed

data.to_csv(sys.argv[2], sep='\t',index=False)
 

    
