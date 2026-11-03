# Test-set results (MIMIC-III, GAMENet split, 1,058 test patients)

Ground-truth DDI rate of the real prescriptions in the training data is shown in the README.

| Model | Jaccard | PR-AUC | F1 | DDI rate | Avg #drugs | Params |
|---|---|---|---|---|---|---|
| Previous-visit copy | 0.4324 | 0.4293 | 0.5924 | 0.0973 | 22.8 | - |
| Logistic Regression | 0.4871 | 0.7535 | 0.6456 | 0.0772 | 17.2 | - |
| GAMENet (reproduced) | 0.4852 | 0.7456 | 0.6442 | 0.0945 | 18.9 | 455,002 |
| GAMENet (reproduced) + DDI | 0.4612 | 0.7302 | 0.6227 | 0.0933 | 16.8 | 455,002 |
| LM-SafeMedRec, no text, DDI loss | 0.5231 | 0.7664 | 0.6780 | 0.0710 | 23.7 | 325,362 |
| LM-SafeMedRec, no text, DDI loss + DDI-aware decoding | 0.5115 | 0.7664 | 0.6679 | 0.0464 | 20.3 | 325,362 |
| LM-SafeMedRec, text=tfidf, DDI loss | 0.5218 | 0.7697 | 0.6765 | 0.0716 | 24.1 | 362,802 |
| LM-SafeMedRec, text=tfidf, DDI loss + DDI-aware decoding | 0.5124 | 0.7697 | 0.6686 | 0.0462 | 20.5 | 362,802 |
