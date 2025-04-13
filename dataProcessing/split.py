import sys
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv(sys.argv[1], header=0, delimiter='\t', dtype=str)
dataText = data[["syllabified", "notSyllabified"]]
dataLabels = data[["numeric"]]

X_train, X_test, y_train, y_test = train_test_split(dataText, dataLabels, test_size=.2)

X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=.5)


pd.concat([X_train, y_train], axis=1).to_csv("train.tsv", sep="\t", index=False)

pd.concat([X_test, y_test], axis=1).to_csv("test.tsv", sep="\t", index=False)

pd.concat([X_dev, y_dev], axis=1).to_csv("dev.tsv", sep="\t", index=False)