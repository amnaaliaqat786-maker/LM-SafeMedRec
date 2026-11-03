"""Collect results/*.json into results/RESULTS.md and results/comparison.png."""
import glob
import json
import os

from data import ROOT

R = os.path.join(ROOT, "results")
LABELS = {
    "previous_visit": "Previous-visit copy",
    "logistic_regression": "Logistic Regression",
}


def label(run):
    a = run["args"]
    if a["model"] == "gamenet":
        return "GAMENet (reproduced)" + (" + DDI" if a["ddi"] else "")
    parts = ["LM-SafeMedRec"]
    parts.append("no text" if a["text"] == "none" else f"text={a['text']}")
    if a.get("no_hist"):
        parts.append("no history path")
    parts.append("DDI loss" if a["ddi_loss"] else "no DDI loss")
    return ", ".join(parts)


rows = []
b = os.path.join(R, "baselines.json")
if os.path.exists(b):
    for k, v in json.load(open(b)).items():
        rows.append((LABELS[k], v, None, "-"))
for f in sorted(glob.glob(os.path.join(R, "*.json"))):
    if f.endswith("baselines.json"):
        continue
    run = json.load(open(f))
    rows.append((label(run), run["test"], run["params"], run["best_epoch"]))
    if "test_ddi_decode" in run:
        rows.append((label(run) + " + DDI-aware decoding", run["test_ddi_decode"], run["params"], run["best_epoch"]))

lines = ["| Model | Jaccard | PR-AUC | F1 | DDI rate | Avg #drugs | Params |",
         "|---|---|---|---|---|---|---|"]
for name, m, p, ep in rows:
    lines.append(f"| {name} | {m['ja']:.4f} | {m['prauc']:.4f} | {m['f1']:.4f} | {m['ddi']:.4f} | "
                 f"{m['avg_meds']:.1f} | {'-' if p is None else f'{p:,}'} |")
table = "\n".join(lines)
with open(os.path.join(R, "RESULTS.md"), "w") as fh:
    fh.write("# Test-set results (MIMIC-III, GAMENet split, 1,058 test patients)\n\n")
    fh.write("Ground-truth DDI rate of the real prescriptions in the training data is shown in the README.\n\n")
    fh.write(table + "\n")
print(table)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, m, _, _ in rows:
        ax.scatter(m["ddi"], m["ja"], s=60)
        short = name.replace("LM-SafeMedRec", "Ours").replace(" + DDI-aware decoding", " + DDI filter")
        below = "no text" in name           # near-identical points: put one label under the dot
        ax.annotate(short, (m["ddi"], m["ja"]), fontsize=7, textcoords="offset points",
                    xytext=(6, -10 if below else 4))
    ax.set_xlabel("DDI rate (lower is safer)")
    ax.set_ylabel("Jaccard (higher is more accurate)")
    ax.set_title("Accuracy vs. safety on the MIMIC-III test set")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "comparison.png"), dpi=160)
except ImportError:
    pass
