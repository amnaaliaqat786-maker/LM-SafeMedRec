"""Two simple reference points, scored with the same metrics and split.

1. Previous visit: recommend exactly the drugs given at the patient's last
   visit (first visit: the 25 most frequent training drugs). Shows how much of
   the task is plain repetition.
2. Logistic Regression: one-vs-rest on the multi-hot diagnoses + procedures of
   the current visit (the "LR" baseline in GAMENet and SafeDrug).
"""
import json
import os
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from data import load_all, split, ROOT
from metrics import evaluate

records, voc, voc_size, ddi, ehr = load_all()
train, val, test = split(records)
n_med = voc_size[2]
results = {}

# --- previous visit
freq = np.zeros(n_med)
for p in train:
    for v in p:
        freq[v[2]] += 1
top = np.argsort(-freq)[:25]


def prev_visit(patient):
    out = np.zeros((len(patient), n_med))
    out[0, top] = 1
    for t in range(1, len(patient)):
        out[t, patient[t - 1][2]] = 1
    return out


results["previous_visit"] = evaluate(prev_visit, test, n_med, ddi)

# --- logistic regression
def featurise(patients):
    rows = [v for p in patients for v in p]
    X = lil_matrix((len(rows), voc_size[0] + voc_size[1]))
    Y = np.zeros((len(rows), n_med))
    for i, v in enumerate(rows):
        X[i, v[0]] = 1
        X[i, [voc_size[0] + c for c in v[1]]] = 1
        Y[i, v[2]] = 1
    return X.tocsr(), Y


Xtr, Ytr = featurise(train)
keep = Ytr.sum(0) > 0               # drugs never seen in training cannot be learned
clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=500), n_jobs=1).fit(Xtr, Ytr[:, keep])


def lr_predict(patient):
    X, _ = featurise([patient])
    out = np.zeros((len(patient), n_med))
    out[:, keep] = clf.predict_proba(X)
    return out


results["logistic_regression"] = evaluate(lr_predict, test, n_med, ddi)

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "baselines.json"), "w") as fh:
    json.dump(results, fh, indent=1)
for k, v in results.items():
    print(k, {m: round(x, 4) for m, x in v.items()})
