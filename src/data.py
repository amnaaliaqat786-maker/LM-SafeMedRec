"""Data loading for LM-SafeMedRec.

Uses the processed MIMIC-III files released with GAMENet
(https://github.com/sjy1203/GAMENet, data/ folder) and the same patient split
as the original paper so results are comparable:
    first 2/3 of patients -> train
    next 1/6              -> test
    last 1/6              -> validation (model selection)

Each patient is a list of visits; each visit is [diag_ids, proc_ids, med_ids].
"""
import os
import re
import numpy as np
import pandas as pd

from safe_load import SafeUnpickler, load as safe_load, load_vocab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMENET_DIR = os.path.join(ROOT, "data", "gamenet")
DESC_DIR = os.path.join(ROOT, "data", "descriptions")


def load_records():
    with open(os.path.join(GAMENET_DIR, "records_final.pkl"), "rb") as fh:
        return SafeUnpickler(fh).load()


def load_all():
    records = load_records()
    voc = load_vocab(os.path.join(GAMENET_DIR, "voc_final.pkl"))
    ddi = np.asarray(safe_load(os.path.join(GAMENET_DIR, "ddi_A_final.pkl")), dtype=np.float32)
    ehr = np.asarray(safe_load(os.path.join(GAMENET_DIR, "ehr_adj_final.pkl")), dtype=np.float32)
    voc_size = (len(voc["diag_voc"]), len(voc["pro_voc"]), len(voc["med_voc"]))
    return records, voc, voc_size, ddi, ehr


def split(records):
    n = int(len(records) * 2 / 3)
    train = records[:n]
    eval_len = int(len(records[n:]) / 2)
    test = records[n:n + eval_len]
    val = records[n + eval_len:]
    return train, val, test


# ---------------------------------------------------------------- code text
def _dx_table():
    df = pd.read_csv(os.path.join(DESC_DIR, "cms32_dx.csv"), dtype=str)
    return dict(zip(df.iloc[:, 0].str.strip(), df.iloc[:, 1].str.strip()))


def _sg_table():
    out = {}
    with open(os.path.join(DESC_DIR, "CMS32_DESC_LONG_SG.txt"), encoding="latin-1") as fh:
        for line in fh:
            code, _, desc = line.rstrip("\n").partition(" ")
            out[code.strip()] = desc.strip()
    return out


def _atc_table():
    df = pd.read_csv(os.path.join(DESC_DIR, "who_atc_2021.csv"), dtype=str)
    return dict(zip(df["atc_code"], df["atc_name"]))


def _lookup_prefix(code, table):
    """Exact match, else the closest parent code (drop trailing digits)."""
    c = code
    while c:
        if c in table:
            return table[c], (c == code)
        # parent level: any code that starts with c
        c = c[:-1]
    return None, False


def code_texts(voc):
    """Return text descriptions aligned with each vocabulary + coverage stats."""
    dx, sg, atc = _dx_table(), _sg_table(), _atc_table()
    # parent lookup helpers: map a prefix to the first matching description
    dx_prefix = {}
    for k, v in dx.items():
        for i in range(3, len(k)):
            dx_prefix.setdefault(k[:i], v)
    texts, stats = {}, {}

    diag = []
    exact = 0
    for code in voc["diag_voc"]:
        if code in dx:
            diag.append(dx[code]); exact += 1
        elif code in dx_prefix:
            diag.append(dx_prefix[code])
        else:
            diag.append(f"diagnosis code {code}")
    texts["diag"], stats["diag_exact"] = diag, exact

    proc = []
    exact = 0
    for code in voc["pro_voc"]:
        cand = [code, code.zfill(4), code.zfill(3)]
        hit = next((sg[c] for c in cand if c in sg), None)
        if hit is None:  # MIMIC drops leading zeros and sometimes the last digit
            hit = next((v for k, v in sg.items() if k.startswith(code.zfill(3))), None)
        else:
            exact += 1
        proc.append(hit or f"procedure code {code}")
    texts["proc"], stats["proc_exact"] = proc, exact

    med = []
    exact = 0
    for code in voc["med_voc"]:
        name, ok = _lookup_prefix(code, atc)
        level3 = atc.get(code[:3], "")
        if name:
            exact += ok
            txt = name.lower()
            if level3 and level3.lower() not in txt:
                txt = f"{level3.lower()}: {txt}"
            med.append(txt)
        else:
            med.append(f"drug class {code}")
    texts["med"], stats["med_exact"] = med, exact

    # the two GAMENet padding tokens
    for key, vkey in (("diag", "diag_voc"), ("proc", "pro_voc"), ("med", "med_voc")):
        for i, code in enumerate(voc[vkey]):
            if code in ("seperator", "decoder_point"):
                texts[key][i] = "[special token]"
    return texts, stats


if __name__ == "__main__":
    records, voc, voc_size, ddi, ehr = load_all()
    tr, va, te = split(records)
    print("vocab sizes (diag, proc, med):", voc_size)
    print("patients train/val/test:", len(tr), len(va), len(te))
    print("visits:", sum(len(p) for p in records))
    print("DDI pairs:", int(ddi.sum() / 2), " EHR co-prescription edges:", int((ehr > 0).sum() / 2))
    t, s = code_texts(voc)
    print("description coverage (exact):", {k: f"{v}/{len(t[k.split('_')[0]])}" for k, v in s.items()})
    for k in t:
        print(k, t[k][2:6])
