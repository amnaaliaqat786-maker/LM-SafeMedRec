# Analysis of `lmsmr_text-tfidf_ddi-1_s0`

## 1. Reliance on text embeddings

Mean gate value g in e = (1 - g) * ID + g * Text (0 = ignores text, 1 = text only).
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

## 3. Privacy Note
Individual patient-level examples, diagnosis histories, medication recommendations, and prescription details are excluded from this public repository to protect the confidentiality of clinical data.

The model's performance is evaluated using aggregate test-set metrics and statistical analyses.
