"""Turn code descriptions into vectors.

Two encoders:
  --encoder tfidf               offline lexical baseline (word + char n-gram TF-IDF -> SVD)
  --encoder hf --model NAME     any Hugging Face model, mean-pooled last hidden state
                                (run this one on Google Colab, where Hugging Face is reachable)

Examples
  python text_embed.py --encoder tfidf
  python text_embed.py --encoder hf --model emilyalsentzer/Bio_ClinicalBERT
  python text_embed.py --encoder hf --model Qwen/Qwen2.5-0.5B-Instruct

Output: data/text_emb/<name>.npz with arrays diag, proc, med (rows aligned to the vocabularies).
"""
import argparse
import os
import numpy as np

from data import load_all, code_texts, ROOT

OUT_DIR = os.path.join(ROOT, "data", "text_emb")


def tfidf_embed(texts, dim=128, seed=0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from scipy.sparse import hstack
    allt = texts["diag"] + texts["proc"] + texts["med"]
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(allt)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True).fit(allt)
    svd = TruncatedSVD(dim, random_state=seed).fit(hstack([w.transform(allt), c.transform(allt)]))
    f = lambda t: svd.transform(hstack([w.transform(t), c.transform(t)]))
    return {k: f(texts[k]).astype(np.float32) for k in ("diag", "proc", "med")}


def hf_embed(texts, model_name, batch=32, prefix=""):
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    out = {}
    for key, kind in (("diag", "diagnosis"), ("proc", "procedure"), ("med", "medication class")):
        rows = []
        items = [f"{prefix}{kind}: {t}" for t in texts[key]]
        for i in range(0, len(items), batch):
            enc = tok(items[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            rows.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
        out[key] = np.concatenate(rows).astype(np.float32)
        print(key, out[key].shape)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["tfidf", "hf"], default="tfidf")
    ap.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--name", default=None, help="output file name (default from encoder/model)")
    a = ap.parse_args()
    _, voc, *_ = load_all()
    texts, stats = code_texts(voc)
    print("description coverage:", stats)
    emb = tfidf_embed(texts) if a.encoder == "tfidf" else hf_embed(texts, a.model)
    name = a.name or ("tfidf" if a.encoder == "tfidf" else a.model.split("/")[-1])
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.npz")
    np.savez(path, **emb)
    print("saved", path)
