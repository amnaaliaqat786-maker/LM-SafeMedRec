"""Evaluation metrics, ported from GAMENet's util.py so numbers are comparable.

Per patient, metrics are averaged over that patient's visits; the reported
score is the mean over patients (exactly as in GAMENet / SafeDrug).
"""
import numpy as np
from sklearn.metrics import average_precision_score


def ddi_rate(med_lists, ddi):
    """Share of drug pairs inside predicted sets that are known interactions."""
    all_cnt = dd_cnt = 0
    for meds in med_lists:
        meds = list(meds)
        for i in range(len(meds)):
            for j in range(i + 1, len(meds)):
                all_cnt += 1
                if ddi[meds[i], meds[j]] == 1 or ddi[meds[j], meds[i]] == 1:
                    dd_cnt += 1
    return 0.0 if all_cnt == 0 else dd_cnt / all_cnt


def patient_metrics(y_gt, y_pred, y_prob):
    """y_* are (n_visits, n_meds) arrays for one patient."""
    ja, prc, rec = [], [], []
    for b in range(y_gt.shape[0]):
        target = set(np.where(y_gt[b] == 1)[0])
        out = set(np.where(y_pred[b] == 1)[0])
        inter, union = target & out, target | out
        ja.append(0 if len(union) == 0 else len(inter) / len(union))
        prc.append(0 if len(out) == 0 else len(inter) / len(out))
        rec.append(0 if len(target) == 0 else len(inter) / len(target))
    f1 = [0 if p + r == 0 else 2 * p * r / (p + r) for p, r in zip(prc, rec)]
    prauc = np.mean([average_precision_score(y_gt[b], y_prob[b]) for b in range(len(y_gt))])
    return np.mean(ja), prauc, np.mean(prc), np.mean(rec), np.mean(f1)


def evaluate(predict_patient, patients, n_med, ddi, threshold=0.5, decode=None):
    """predict_patient(patient) -> (n_visits, n_med) probabilities.

    decode: optional function(prob_vector, label_list) -> label_list applied
    after thresholding (used for DDI-aware decoding).
    """
    scores = {k: [] for k in ("ja", "prauc", "p", "r", "f1")}
    all_sets, n_meds = [], []
    for patient in patients:
        prob = predict_patient(patient)
        gt = np.zeros((len(patient), n_med))
        pred = np.zeros_like(gt)
        for v, visit in enumerate(patient):
            gt[v, visit[2]] = 1
            labels = list(np.where(prob[v] >= threshold)[0])
            if decode is not None:
                labels = decode(prob[v], labels)
            pred[v, labels] = 1
            all_sets.append(labels)
            n_meds.append(len(labels))
        ja, prauc, p, r, f1 = patient_metrics(gt, pred, prob)
        for k, val in zip(("ja", "prauc", "p", "r", "f1"), (ja, prauc, p, r, f1)):
            scores[k].append(val)
    out = {k: float(np.mean(v)) for k, v in scores.items()}
    out["ddi"] = ddi_rate(all_sets, ddi)
    out["avg_meds"] = float(np.mean(n_meds))
    return out


def ddi_aware_decode(ddi, margin=0.0):
    """Greedy safety filter: walk the predicted drugs from most to least
    confident and skip a drug if it interacts with one already kept, unless its
    probability is above (0.5 + margin) higher-confidence override.

    margin = 1.0 means never override (strict filter)."""
    def decode(prob, labels):
        kept = []
        for d in sorted(labels, key=lambda x: -prob[x]):
            clash = any(ddi[d, k] == 1 or ddi[k, d] == 1 for k in kept)
            if clash and prob[d] < 0.5 + margin:
                continue
            kept.append(d)
        return kept
    return decode
