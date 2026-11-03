"""Train and evaluate GAMENet (reproduced) or LM-SafeMedRec.

  python train.py --model gamenet --ddi --epochs 15
  python train.py --model lmsmr --text none
  python train.py --model lmsmr --text tfidf --ddi_loss
  python train.py --model lmsmr --text Bio_ClinicalBERT --ddi_loss   # after text_embed.py on Colab

Best epoch is chosen by validation Jaccard; the test set is scored once with
that checkpoint. Results go to results/<run_name>.json.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from data import load_all, split, ROOT
from metrics import evaluate, ddi_rate, ddi_aware_decode
from gamenet import GAMENet
from lm_safemedrec import LMSafeMedRec

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["gamenet", "lmsmr"], default="lmsmr")
ap.add_argument("--text", default="none", help="none | tfidf | name of a file in data/text_emb")
ap.add_argument("--ddi", action="store_true", help="GAMENet: DDI memory + annealed DDI loss")
ap.add_argument("--ddi_loss", action="store_true", help="LM-SafeMedRec: SafeDrug-style controllable DDI loss")
ap.add_argument("--target_ddi", type=float, default=0.06)
ap.add_argument("--kp", type=float, default=0.05)
ap.add_argument("--no_hist", action="store_true", help="ablation: drop history-copy path")
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--name", default=None)
ap.add_argument("--threads", type=int, default=2)
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(args.threads)

records, voc, voc_size, ddi, ehr = load_all()
train, val, test = split(records)
n_med = voc_size[2]

text = None
if args.model == "lmsmr" and args.text != "none":
    z = np.load(os.path.join(ROOT, "data", "text_emb", f"{args.text}.npz"))
    text = {k: z[k] for k in ("diag", "proc", "med")}

if args.model == "gamenet":
    model = GAMENet(voc_size, ehr, ddi, emb_dim=64, ddi_in_memory=args.ddi)
    lr = args.lr or 2e-4
else:
    model = LMSafeMedRec(voc_size, ehr, ddi, text=text, dim=args.dim, layers=args.layers)
    if args.no_hist:
        model.hist_w.data.zero_()
        model.hist_w.requires_grad_(False)
    lr = args.lr or 5e-4
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

name = args.name or f"{args.model}_text-{args.text}_ddi-{int(args.ddi or args.ddi_loss)}_s{args.seed}"
print(f"{name}: {n_params:,} trainable parameters")


def margin_target(meds):
    t = torch.full((1, n_med), -1, dtype=torch.long)
    t[0, :len(meds)] = torch.tensor(meds)
    return t


def predict(patient):
    model.eval()
    return model.forward_all(patient)


# GAMENet's original DDI annealing settings
T, decay, TARGET = 0.5, 0.85, 0.05
history, best, best_state = [], -1, None
for epoch in range(args.epochs):
    t0 = time.time()
    model.train()
    losses = []
    order = np.random.permutation(len(train))
    for pi in order:
        patient = train[pi]
        if args.model == "gamenet":
            for t in range(len(patient)):
                logits, neg = model(patient[:t + 1])
                target = torch.zeros(1, n_med); target[0, patient[t][2]] = 1
                l1 = F.binary_cross_entropy_with_logits(logits, target)
                l3 = F.multilabel_margin_loss(torch.sigmoid(logits), margin_target(patient[t][2]))
                loss = 0.9 * l1 + 0.01 * l3
                if args.ddi:
                    pred = np.where(torch.sigmoid(logits).detach().numpy()[0] >= 0.5)[0]
                    cur = ddi_rate([pred], ddi)
                    if cur > TARGET and np.random.rand() < np.exp((TARGET - cur) / T):
                        loss = neg
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
        else:
            logits = model(patient)
            target = torch.zeros(len(patient), n_med)
            mt = torch.full((len(patient), n_med), -1, dtype=torch.long)
            for t, v in enumerate(patient):
                target[t, v[2]] = 1
                mt[t, :len(v[2])] = torch.tensor(v[2])
            l_multi = 0.95 * F.binary_cross_entropy_with_logits(logits, target) + \
                0.05 * F.multilabel_margin_loss(torch.sigmoid(logits), mt)
            loss = l_multi
            if args.ddi_loss:
                pred = [np.where(r >= 0.5)[0] for r in torch.sigmoid(logits).detach().numpy()]
                cur = ddi_rate(pred, ddi)
                if cur > args.target_ddi:
                    beta = max(0.0, 1 + (args.target_ddi - cur) / args.kp)
                    loss = beta * l_multi + (1 - beta) * model.ddi_loss(logits)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
    if args.model == "gamenet":
        T *= decay
    res = evaluate(predict, val, n_med, ddi)
    res.update(epoch=epoch, loss=float(np.mean(losses)), minutes=(time.time() - t0) / 60)
    history.append(res)
    print(f"ep {epoch:2d} loss {res['loss']:.4f} | val ja {res['ja']:.4f} prauc {res['prauc']:.4f} "
          f"f1 {res['f1']:.4f} ddi {res['ddi']:.4f} meds {res['avg_meds']:.1f} | {res['minutes']:.1f} min", flush=True)
    if res["ja"] > best:
        best = res["ja"]
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
best_ep = int(np.argmax([h["ja"] for h in history]))
test_res = evaluate(predict, test, n_med, ddi)
out = {"name": name, "args": vars(args), "params": n_params, "best_epoch": best_ep,
       "val": history[best_ep], "test": test_res, "history": history}
if args.model == "lmsmr":
    out["test_ddi_decode"] = evaluate(predict, test, n_med, ddi, decode=ddi_aware_decode(ddi, margin=0.3))
    os.makedirs(os.path.join(ROOT, "results", "checkpoints"), exist_ok=True)
    torch.save(best_state, os.path.join(ROOT, "results", "checkpoints", f"{name}.pt"))
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", f"{name}.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("TEST", json.dumps({k: round(v, 4) for k, v in test_res.items()}))
if "test_ddi_decode" in out:
    print("TEST+decode", json.dumps({k: round(v, 4) for k, v in out["test_ddi_decode"].items()}))
