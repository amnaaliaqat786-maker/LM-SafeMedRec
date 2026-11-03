# Analysis of `lmsmr_text-tfidf_ddi-1_s0`

## 1. Reliance on text embeddings

Mean gate value g in e = (1-g)·ID + g·Text (0 = ignores text, 1 = text only).

| Code type | mean g |
|---|---|
| Diagnosis | 0.260 |
| Procedure | 0.290 |
| Drug class | 0.292 |

## 2. Accuracy by diagnosis rarity (test visits)

Each visit is grouped by the training frequency of its rarest diagnosis code.

| Group | Visits | Jaccard (this run) | Jaccard (`lmsmr_text-none_ddi-1_s0`) | Difference |
|---|---|---|---|---|
| rarest diagnosis seen < 10 times | 1012 | 0.5270 | 0.5232 | +0.0037 |
| 10 to 99 times | 1181 | 0.5240 | 0.5273 | -0.0032 |
| 100+ times | 71 | 0.5217 | 0.5309 | -0.0092 |

## 3. Case study: test patient 1, visit 3 of 3

Diagnoses at this visit: Calculus of kidney; Multiple sclerosis; Neurogenic bladder NOS; Other specified cardiac dysrhythmias; Personal history, urinary (tract) infection

Attention of the history path over earlier visits:

- visit 1: weight 0.06 (diagnoses: Urinary tract infection, site not specified; Grand mal status; Pseudomonas infection in conditions classified elsewhere and of unspecified site)
- visit 2: weight 0.94 (diagnoses: Calculus of kidney; Septicemia due to pseudomonas; Urinary tract infection, site not specified)

Top recommended drug classes:

| ATC | Drug class | Probability | Actually given | Kept after DDI filter |
|---|---|---|---|---|
| N02B | other analgesics and antipyretics | 0.99 | yes | yes |
| A06A | drugs for constipation | 0.99 | yes | yes |
| A12C | other mineral supplements | 0.99 | yes | yes |
| B01A | antithrombotic agents | 0.99 | yes | yes |
| J01D | antibacterials for systemic use: other beta-lactam antibacterials | 0.97 | yes | yes |
| A02B | drugs for acid related disorders: drugs for peptic ulcer and gastro-oesophageal reflux disease (gord) | 0.97 | yes | yes |
| A04A | antiemetics and antinauseants | 0.96 | yes | yes |
| N02A | analgesics: opioids | 0.95 | yes | yes |
| B05C | blood substitutes and perfusion solutions: irrigating solutions | 0.95 | no | yes |
| A07A | antidiarrheals, intestinal antiinflammatory/antiinfective agents: intestinal antiinfectives | 0.89 | yes | yes |
| A01A | stomatological preparations | 0.87 | no | yes |
| A12B | mineral supplements: potassium | 0.86 | yes | yes |

Interacting pairs in the raw recommendation: 8: N02B–M03B, A01A–N03A, A01A–M03B, A02B–M03B, N02A–B01A, N02A–N03A, B01A–M03B, N03A–D04A

Jaccard for this visit: 0.706 raw, 0.688 after the DDI filter.
