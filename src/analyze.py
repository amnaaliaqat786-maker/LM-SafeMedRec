"""Aggregate interpretability and error analysis for LM-SafeMedRec.

Usage:
    python analyze.py --run lmsmr_text-tfidf_ddi-1_s0
    python analyze.py --run lmsmr_text-tfidf_ddi-1_s0 \
        --compare lmsmr_text-none_ddi-1_s0

Writes:
    results/analysis_<run>.md

The report contains:
    1. Aggregate text-versus-ID embedding gate values.
    2. Aggregate Jaccard scores grouped by diagnosis rarity.

Privacy:
    This script does not export individual patient records,
    diagnosis histories, medication recommendations, or
    patient-level case studies.
"""

import argparse
import json
import os

import numpy as np
import torch

from data import load_all, split, ROOT
from lm_safemedrec import LMSafeMedRec


# --------------------------------------------------
# 1. Command-line arguments
# --------------------------------------------------

ap = argparse.ArgumentParser(
    description="Aggregate interpretability and error analysis."
)

ap.add_argument("--run", required=True)
ap.add_argument("--compare", default=None)

a = ap.parse_args()


# --------------------------------------------------
# 2. Load dataset and split
# --------------------------------------------------

records, voc, voc_size, ddi, ehr = load_all()

train, val, test = split(records)


# --------------------------------------------------
# 3. Load trained model
# --------------------------------------------------

def load_model(run):

    result_path = os.path.join(
        ROOT, "results", f"{run}.json"
    )

    with open(result_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)["args"]

    text = None

    if cfg["text"] != "none":

        embedding_path = os.path.join(
            ROOT,
            "data",
            "text_emb",
            f"{cfg['text']}.npz"
        )

        z = np.load(embedding_path)

        text = {
            k: z[k]
            for k in ("diag", "proc", "med")
        }

    model = LMSafeMedRec(
        voc_size,
        ehr,
        ddi,
        text=text,
        dim=cfg["dim"],
        layers=cfg["layers"]
    )

    checkpoint_path = os.path.join(
        ROOT,
        "results",
        "checkpoints",
        f"{run}.pt"
    )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )

    return model.eval()


model = load_model(a.run)


# --------------------------------------------------
# 4. Initialize aggregate report
# --------------------------------------------------

out = [
    f"# Aggregate Analysis of `{a.run}`\n"
]


# --------------------------------------------------
# 5. Text embedding gate analysis
# --------------------------------------------------

out.append("## 1. Reliance on text embeddings\n")

out.append(
    "Mean gate value g in e = (1-g) * ID + g * Text "
    "(0 = ignores text, 1 = text only).\n"
)

out.append("| Code type | Mean gate value |")
out.append("|---|---|")

for name, module in (
    ("Diagnosis", model.diag),
    ("Procedure", model.proc),
    ("Drug class", model.drug)
):

    if module.use_text:

        gate_value = (
            torch.sigmoid(module.gate).mean().item()
        )

    else:

        gate_value = 0.0

    out.append(
        f"| {name} | {gate_value:.3f} |"
    )


# --------------------------------------------------
# 6. Diagnosis rarity analysis
# --------------------------------------------------

freq = np.zeros(voc_size[0])

for patient in train:

    for visit in patient:

        for diagnosis in visit[0]:

            freq[diagnosis] += 1


def visit_jaccards(model):

    results = []

    with torch.no_grad():

        for patient in test:

            probabilities = model.forward_all(patient)

            for t, visit in enumerate(patient):

                predictions = set(
                    np.where(probabilities[t] >= 0.5)[0]
                )

                actual = set(visit[2])

                union = predictions | actual

                if union:

                    score = (
                        len(predictions & actual)
                        / len(union)
                    )

                else:

                    score = 0.0

                if len(visit[0]) == 0:
                    rarest_frequency = 0
                else:
                    rarest_frequency = min(
                        freq[diagnosis]
                        for diagnosis in visit[0]
                    )
                results.append(
                    (rarest_frequency, score)
                )

    return results

# Calculate results for the current model

current_results = visit_jaccards(model)

# Optional comparison model

comparison_results = None

if a.compare:

    comparison_model = load_model(a.compare)

    comparison_results = visit_jaccards(
        comparison_model
    )

    if len(comparison_results) != len(current_results):
        raise ValueError(
            "Current and comparison models have different "
            "numbers of test visits."
        )
# --------------------------------------------------
# 7. Group results by diagnosis rarity
# --------------------------------------------------

bins = [
    (0, 10, "Fewer than 10 training occurrences"),
    (10, 100, "10 to 99 times"),
    (100, float("inf"), "100+ times")
]

out.append(
    "\n## 2. Accuracy by diagnosis rarity (aggregate test results)\n"
)

out.append(
    "Each test visit is grouped by the training frequency "
    "of its rarest diagnosis code. Visits without diagnosis "
    "codes are included in the first group.\n"
)

if comparison_results is not None:

    out.append(
        "| Group | Visits | Jaccard (current) | "
        "Jaccard (comparison) | Difference |"
    )

    out.append(
        "|---|---:|---:|---:|---:|"
    )

else:

    out.append(
        "| Group | Visits | Jaccard (current) |"
    )

    out.append(
        "|---|---:|---:|"
    )


for lower, upper, label in bins:

    indices = [
        i
        for i, (frequency, score) in enumerate(current_results)
        if lower <= frequency < upper
    ]

    if not indices:

        current_jaccard = float("nan")

    else:

        current_jaccard = np.mean([
            current_results[i][1]
            for i in indices
        ])

    if comparison_results is not None:

        if indices:

            comparison_jaccard = np.mean([
                comparison_results[i][1]
                for i in indices
            ])

            difference = (
                current_jaccard - comparison_jaccard
            )

        else:

            comparison_jaccard = float("nan")
            difference = float("nan")

        out.append(
            f"| {label} | {len(indices)} | "
            f"{current_jaccard:.4f} | "
            f"{comparison_jaccard:.4f} | "
            f"{difference:+.4f} |"
        )

    else:

        out.append(
            f"| {label} | {len(indices)} | "
            f"{current_jaccard:.4f} |"
        )
# --------------------------------------------------
# 8. Privacy statement
# --------------------------------------------------

out.append(
    "\n## 3. Privacy Note\n"
)

out.append(
    "This report contains aggregate model interpretability "
    "and evaluation statistics only. Individual patient records, "
    "diagnosis histories, medication recommendations, and "
    "patient-level case studies are not exported.\n"
)


# --------------------------------------------------
# 9. Save aggregate report
# --------------------------------------------------

output_path = os.path.join(
    ROOT,
    "results",
    f"analysis_{a.run}.md"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(out))


print("\n".join(out))

print(
    f"\nAggregate analysis saved to: {output_path}"
)