#in-file, #outfile
import sys
import re 
import pandas as pd
import numpy as np

syllabified = pd.read_csv(sys.argv[1], header=0, delimiter='\t')
syllabified['numeric'] = np.zeros(syllabified.shape[0],)

for index, row in syllabified.iterrows():
    start = row['syllabified']
    boundaries = re.sub(r';[a-zA-Z]','1', str(start))
    done = re.sub(r'[^1]', '0', boundaries)
    syllabified.at[index,'numeric'] = done

syllabified.to_csv(sys.argv[2], sep='\t',index=False)
    

