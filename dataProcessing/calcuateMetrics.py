import numpy as np

cm = np.array([[14155, 313],
                [308, 3418]])

tn = cm[0, 0]
fp = cm[0, 1]
fn = cm[1, 0]
tp = cm[1, 1]

# Metrics for class 0 (treated as positive)
p1 = tn / (tn + fn) if (tn + fn) > 0 else 0
r1 = tn / (tn + fp) if (tn + fp) > 0 else 0
s1 = tn + fp

# Metrics for class 1 (treated as positive)
p2 = tp / (tp + fp) if (tp + fp) > 0 else 0
r2 = tp / (tp + fn) if (tp + fn) > 0 else 0
s2 = fn + tp

f1 = 2 * (p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0
f2 = 2 * (p2 * r2) / (p2 + r2) if (p2 + r2) > 0 else 0

# Accuracy
a = (tn + tp) / (tn + tp + fn + fp)

# Macro avg
macP = (p1 + p2) / 2
macR = (r1 + r2) / 2
macF = (f1 + f2) / 2

# Micro avg (weighted avg)
totalS = s1 + s2
micP = (p1 * s1 + p2 * s2) / totalS
micR = (r1 * s1 + r2 * s2) / totalS
micF = (f1 * s1 + f2 * s2) / totalS

print(f"\t\tprec\trec\tf1\tsupport\n")
print(f"0\t\t{p1:.4f}\t{r1:.4f}\t{f1:.4f}\t{s1}\n")
print(f"1\t\t{p2:.4f}\t{r2:.4f}\t{f2:.4f}\t{s2}\n")
print(f"accuracy\t\t\t{a:.4f}\t{totalS}\n")
print(f"macro avg\t{macP:.4f}\t{macR:.4f}\t{macF:.4f}\t{totalS}\n")
print(f"weighted avg\t{micP:.4f}\t{micR:.4f}\t{micF:.4f}\t{totalS}\n")