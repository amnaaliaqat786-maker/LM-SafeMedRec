"""Interpretability and error analysis for a trained LM-SafeMedRec checkpoint.

  python analyze.py --run lmsmr_text-tfidf_ddi-1_s0 --compare lmsmr_text-none_ddi-1_s0

Writes results/analysis_<run>.md with:
  1. how much the model leans on text vs. learned ID embeddings (gate values)
  2. Jaccard on test visits grouped by how rare their diagnoses are in training,
     compared with the --compare run (does text help rare codes?)
  3. a case study: one test patient, the drugs recommended at the last visit,
     which earlier visits the history path attended to, and any interacting pairs
"""
import argparse
import json
import os
import numpy as np
import torch

from data import load_all, split, code_texts, ROOT
from lm_safemedrec import LMSafeMedRec
from metrics import ddi_aware_decode

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--compare", default=None)
ap.add_argument("--patient", type=int, default=None, help="index into test set for the case study")
a = ap.parse_args()

records, voc, voc_size, ddi, ehr = load_all()
train, val, test = split(records)
texts, _ = code_texts(voc)
n_med = voc_size[2]


def load_model(run):
    cfg = json.load(open(os.path.join(ROOT, "results", f"{run}.json")))["args"]
    text = None
    if cfg["text"] != "none":
        z = np.load(os.path.join(ROOT, "data", "text_emb", f"{cfg['text']}.npz"))
        text = {k: z[k] for k in ("diag", "proc", "med")}
    m = LMSafeMedRec(voc_size, ehr, ddi, text=text, dim=cfg["dim"], layers=cfg["layers"])
    m.load_state_dict(torch.load(os.path.join(ROOT, "results", "checkpoints", f"{run}.pt")))
    return m.eval()


model = load_model(a.run)
out = [f"# Analysis of `{a.run}`\n"]

# 1. gates
out.append("## 1. Reliance on text embeddings\n")
out.append("Mean gate value g in e = (1-g)·ID + g·Text (0 = ignores text, 1 = text only).\n")
out.append("| Code type | mean g |\n|---|---|")
for nm, mod in (("Diagnosis", model.diag), ("Procedure", model.proc), ("Drug class", model.drug)):
    g = torch.sigmoid(mod.gate).mean().item() if mod.use_text else 0.0
    out.append(f"| {nm} | {g:.3f} |")

# 2. rare-code analysis
freq = np.zeros(voc_size[0])
for p in train:
    for v in p:
        freq[v[0]] += 1


def visit_jaccards(m):
    res = []
    for p in test:
        prob = m.forward_all(p)
        for t, v in enumerate(p):
            pred = set(np.where(prob[t] >= 0.5)[0]); gt = set(v[2])
            u = pred | gt
            res.append((min(freq[v[0]]), len(pred & gt) / len(u) if u else 0))
    return res


bins = [(0, 10, "rarest diagnosis seen < 10 times"), (10, 100, "10 to 99 times"), (100, 1e9, "100+ times")]
cur = visit_jaccards(model)
cmp = visit_jaccards(load_model(a.compare)) if a.compare else None
out.append("\n## 2. Accuracy by diagnosis rarity (test visits)\n")
out.append("Each visit is grouped by the training frequency of its rarest diagnosis code.\n")
hdr = "| Group | Visits | Jaccard (this run) |" + (f" Jaccard (`{a.compare}`) | Difference |" if cmp else "")
out.append(hdr)
out.append("|---|---|---|" + ("---|---|" if cmp else ""))
for lo, hi, nm in bins:
    idx = [i for i, (f, _) in enumerate(cur) if lo <= f < hi]
    j1 = np.mean([cur[i][1] for i in idx])
    row = f"| {nm} | {len(idx)} | {j1:.4f} |"
    if cmp:
        j2 = np.mean([cmp[i][1] for i in idx])
        row += f" {j2:.4f} | {j1 - j2:+.4f} |"
    out.append(row)

# 3. case study
med_names = texts["med"]
med_codes = voc["med_voc"]
cands = [i for i, p in enumerate(test) if 3 <= len(p) <= 5]
pi = a.patient if a.patient is not None else cands[0]
p = test[pi]
with torch.no_grad():
    logits, attn = model(p, return_attn=True)
prob = torch.sigmoid(logits).numpy()
t = len(p) - 1
pred = list(np.where(prob[t] >= 0.5)[0])
safe = ddi_aware_decode(ddi, margin=0.3)(prob[t], pred)
gt = set(p[t][2])
out.append(f"\n## 3. Case study: test patient {pi}, visit {t + 1} of {len(p)}\n")
out.append("Diagnoses at this visit: " + "; ".join(texts["diag"][c] for c in p[t][0][:6]) +
           (" ..." if len(p[t][0]) > 6 else "") + "\n")
out.append("Attention of the history path over earlier visits:\n")
for i in range(t):
    w = attn[t - 1, i].item()
    out.append(f"- visit {i + 1}: weight {w:.2f} (diagnoses: " +
               "; ".join(texts["diag"][c] for c in p[i][0][:3]) + ")")
top = sorted(pred, key=lambda d: -prob[t, d])[:12]
out.append("\nTop recommended drug classes:\n")
out.append("| ATC | Drug class | Probability | Actually given | Kept after DDI filter |\n|---|---|---|---|---|")
for d in top:
    out.append(f"| {med_codes[d]} | {med_names[d]} | {prob[t, d]:.2f} | {'yes' if d in gt else 'no'} | "
               f"{'yes' if d in safe else 'removed'} |")
pairs = [(med_codes[i], med_codes[j]) for k, i in enumerate(pred) for j in pred[k + 1:] if ddi[i, j] or ddi[j, i]]
out.append(f"\nInteracting pairs in the raw recommendation: {len(pairs)}"
           + (": " + ", ".join(f"{x}–{y}" for x, y in pairs[:8]) if pairs else "") + "\n")
pj = len(set(pred) & gt) / len(set(pred) | gt)
sj = len(set(safe) & gt) / len(set(safe) | gt)
out.append(f"Jaccard for this visit: {pj:.3f} raw, {sj:.3f} after the DDI filter.\n")

path = os.path.join(ROOT, "results", f"analysis_{a.run}.md")
open(path, "w").write("\n".join(out))
print("\n".join(out))
