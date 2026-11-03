# LM-SafeMedRec

Safe medication recommendation from hospital records, using a Transformer over a patient's visit history and language-model embeddings of medical code descriptions.

Given a patient's diagnoses and procedures at the current hospital visit, and their earlier visits, the model recommends a set of drug classes (ATC level 4). It is scored on how close the set is to what doctors actually prescribed (Jaccard, F1, PR-AUC) and on how many recommended drug pairs are known to interact (DDI rate, lower is safer).

## Starting point

This project builds on **GAMENet** (Shang et al., *GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination*, AAAI 2019, [code](https://github.com/sjy1203/GAMENet)). I use its processed MIMIC-III data and its patient split, and I ported its model and metrics to current PyTorch so the comparison is like for like. The DDI loss controller follows **SafeDrug** (Yang et al., IJCAI 2021, [code](https://github.com/ycq091044/SafeDrug)).

## What I changed

| | GAMENet (original) | LM-SafeMedRec (this project) |
|---|---|---|
| Visit history encoder | Two GRUs | Causal Transformer (2 layers, 4 heads, visit-position embeddings) |
| Code embeddings | Learned ID only | Learned ID + gated projection of a text embedding of the code's description |
| Drug representation | GCN over co-prescription and DDI graphs | Drug ID + text embedding, refined over the co-prescription graph |
| Use of past prescriptions | Memory read weighted by visit similarity | Explicit attention over past visits with a learned weight per drug (inspectable) |
| Safety | DDI memory + annealed DDI loss | SafeDrug-style controllable DDI loss + optional DDI-aware decoding at test time |
| Training | One optimizer step per visit prefix | One step per patient, all visits predicted at once (causal masking) |

The text embeddings are the main idea. A diagnosis code like `4280` is just an index to GAMENet, so a code seen only a handful of times in training gets a poor embedding. Its description ("Congestive heart failure, unspecified") carries meaning a language model already knows, and similar descriptions land close together. The gate lets the model learn, per dimension, how much to trust text versus the learned ID vector.

Text sources: ICD-9-CM v32 diagnosis and procedure descriptions (CMS) and WHO ATC names. Coverage is 1,903 of 1,958 diagnosis codes and 1,419 of 1,430 procedure codes matched exactly (the rest fall back to their parent category), and all 151 drug classes.

## Data

MIMIC-III as processed by GAMENet: 6,350 patients, 15,032 visits, 1,958 diagnosis codes, 1,430 procedure codes, 151 drug classes, 467 interacting drug-class pairs (from TWOSIDES). Patients average 2.37 visits and 19.9 drugs per visit. Split: 4,233 train, 1,059 validation, 1,058 test. The real prescriptions have a DDI rate of 0.0845 in training and 0.0750 in test.

MIMIC-III is credentialed data. For any publication, get your own access through PhysioNet and regenerate the files with GAMENet's preprocessing.

## Results

Test set, 1,058 patients, single run (seed 0), 12 epochs, best epoch chosen on validation Jaccard. Produced by `src/summarize.py`.

| Model | Jaccard | PR-AUC | F1 | DDI rate | Avg #drugs | Params |
|---|---|---|---|---|---|---|
| Previous-visit copy | 0.4324 | 0.4293 | 0.5924 | 0.0973 | 22.8 | - |
| Logistic Regression | 0.4871 | 0.7535 | 0.6456 | 0.0772 | 17.2 | - |
| GAMENet (reproduced) | 0.4852 | 0.7456 | 0.6442 | 0.0945 | 18.9 | 455,002 |
| GAMENet (reproduced) + DDI | 0.4612 | 0.7302 | 0.6227 | 0.0933 | 16.8 | 455,002 |
| LM-SafeMedRec, no text, DDI loss | 0.5231 | 0.7664 | 0.6780 | 0.0710 | 23.7 | 325,362 |
| LM-SafeMedRec, no text, DDI loss + DDI filter | 0.5115 | 0.7664 | 0.6679 | 0.0464 | 20.3 | 325,362 |
| LM-SafeMedRec, TF-IDF text, DDI loss | 0.5218 | 0.7697 | 0.6765 | 0.0716 | 24.1 | 362,802 |
| LM-SafeMedRec, TF-IDF text, DDI loss + DDI filter | 0.5124 | 0.7697 | 0.6686 | 0.0462 | 20.5 | 362,802 |
| LM-SafeMedRec, Bio_ClinicalBERT text, DDI loss | run on Colab | | | | | |
| LM-SafeMedRec, Qwen2.5-0.5B text, DDI loss | run on Colab | | | | | |

![Accuracy vs safety](results/comparison.png)

What the numbers show:

- The Transformer model beats both reproduced GAMENet variants and logistic regression by about 0.04 Jaccard, and its DDI rate (0.071) is below that of the real test prescriptions (0.075).
- The DDI filter cuts interacting pairs by a third (0.071 to 0.046) for a 0.012 Jaccard cost. That trade-off is the main safety result.
- TF-IDF text features did not help: 0.5218 with text against 0.5231 without, a difference smaller than run-to-run noise. On visits with very rare diagnoses (seen fewer than 10 times in training) text added +0.004, also within noise. The model does use the text side (its gates give text about 26 to 29% of each embedding), but word overlap is not enough. Language-model embeddings, which encode medical meaning, are the real test of the idea.
- My GAMENet reproduction is weaker than the paper (about 0.52 Jaccard after 40 epochs on GPU). With 12 epochs on CPU its validation score was still rising, and the DDI annealing spends many early steps on the safety loss only. Treat the GAMENet rows as a lower bound.

Details, gate values and a worked patient example are in [`results/analysis_lmsmr_text-tfidf_ddi-1_s0.md`](results/analysis_lmsmr_text-tfidf_ddi-1_s0.md).

## How to run

```bash
pip install -r requirements.txt
bash scripts/get_data.sh                          # downloads data into data/
cd src
python baselines.py                               # previous-visit and logistic regression
python train.py --model gamenet --ddi --epochs 12 # reproduced GAMENet
python text_embed.py --encoder tfidf              # offline text features
python train.py --model lmsmr --text tfidf --ddi_loss --epochs 12
python summarize.py                               # results/RESULTS.md and comparison.png
python analyze.py --run lmsmr_text-tfidf_ddi-1_s0 --compare lmsmr_text-none_ddi-1_s0
```

Language-model embeddings need Hugging Face, so run [`notebooks/colab_lm_embeddings.ipynb`](notebooks/colab_lm_embeddings.ipynb) on Google Colab. It embeds the descriptions with Bio_ClinicalBERT and Qwen2.5-0.5B and trains a model with each.

## Files

```
src/data.py            loading, GAMENet split, code descriptions
src/safe_load.py       reads the GAMENet pickles without executing code stored in them
src/metrics.py         Jaccard, PR-AUC, F1, DDI rate (ported from GAMENet), DDI-aware decoding
src/gamenet.py         GAMENet ported to PyTorch 2
src/lm_safemedrec.py   the new model
src/text_embed.py      TF-IDF or Hugging Face embeddings of code descriptions
src/train.py           training and evaluation for both models
src/baselines.py       previous-visit and logistic regression baselines
src/analyze.py         gate values, rare-code analysis, patient case study
src/summarize.py       results table and figure
```

## Limitations

- Results are from single runs (seed 0) on CPU with 12 epochs; the original papers train longer and average several runs. Differences under about 0.005 Jaccard should not be read as real.
- TF-IDF features capture wording, not medical meaning. The language-model runs are the real test of the idea.
- Drug classes are ATC level 4, not individual drugs or doses, as in GAMENet and SafeDrug.
- The DDI-aware decoder removes drugs by confidence only. It knows nothing about which drug in a pair is clinically more important.
