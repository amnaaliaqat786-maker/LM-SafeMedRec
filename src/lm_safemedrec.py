"""LM-SafeMedRec: the model added in this project.

Differences from GAMENet:
1. Visit history is encoded with a causal Transformer (multi-head self-attention
   with learned visit-position embeddings) instead of two GRUs.
2. Every diagnosis, procedure and drug embedding = learned ID embedding
   + gated projection of a frozen text embedding of the code's description
   (from a language model, or TF-IDF/SVD when no LM is available). This lets
   rare codes borrow meaning from similar descriptions.
3. Drug representations are refined with one propagation step over the EHR
   co-prescription graph and scored against the patient state.
4. A history-copy path: attention over earlier visits returns the drugs given
   then, with a learned per-drug weight (same intuition as GAMENet's dynamic
   memory, but explicit and inspectable).
Training uses SafeDrug's controllable DDI loss (see train.py).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeEmbedding(nn.Module):
    def __init__(self, n, dim, text=None, text_dropout=0.1):
        super().__init__()
        self.id_emb = nn.Embedding(n, dim)
        nn.init.normal_(self.id_emb.weight, std=0.05)
        self.use_text = text is not None
        if self.use_text:
            t = torch.tensor(text, dtype=torch.float32)
            t = (t - t.mean(0)) / (t.std(0) + 1e-6)          # standardise each feature
            self.register_buffer("text", t)
            self.proj = nn.Sequential(nn.Linear(t.shape[1], dim), nn.Tanh(), nn.Linear(dim, dim))
            self.gate = nn.Parameter(torch.zeros(dim))         # starts as 0.5 mix
            self.tdrop = nn.Dropout(text_dropout)

    def table(self):
        e = self.id_emb.weight
        if self.use_text:
            g = torch.sigmoid(self.gate)
            e = (1 - g) * e + g * self.proj(self.tdrop(self.text))
        return e


class LMSafeMedRec(nn.Module):
    def __init__(self, voc_size, ehr_adj, ddi_adj, text=None, dim=64, layers=2, heads=4,
                 dropout=0.3, max_visits=32):
        super().__init__()
        text = text or {}
        self.n_med = voc_size[2]
        self.diag = CodeEmbedding(voc_size[0], dim, text.get("diag"))
        self.proc = CodeEmbedding(voc_size[1], dim, text.get("proc"))
        self.drug = CodeEmbedding(voc_size[2], dim, text.get("med"))
        self.visit_proj = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))
        self.pos = nn.Embedding(max_visits, dim)
        self.max_visits = max_visits
        enc_layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True,
                                               norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.state = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        # history copy path
        self.hist_q = nn.Linear(dim, dim)
        self.hist_k = nn.Linear(dim, dim)
        self.hist_w = nn.Parameter(torch.full((voc_size[2],), 2.0))
        self.bias = nn.Parameter(torch.zeros(voc_size[2]))
        # normalised co-prescription graph for drug refinement
        a = ehr_adj + np.eye(ehr_adj.shape[0])
        d = 1.0 / np.sqrt(a.sum(1))
        self.register_buffer("ehr", torch.tensor(d[:, None] * a * d[None, :], dtype=torch.float32))
        self.register_buffer("ddi_adj", torch.tensor(ddi_adj, dtype=torch.float32))
        self.graph_lin = nn.Linear(dim, dim)

    def drug_table(self):
        e = self.drug.table()
        return e + self.graph_lin(self.ehr @ e)

    def encode(self, patient):
        dev = self.bias.device
        dt, pt = self.diag.table(), self.proc.table()
        vis = []
        for v in patient:
            dvec = self.drop(dt[torch.tensor(v[0], device=dev)]).mean(0)
            pvec = self.drop(pt[torch.tensor(v[1], device=dev)]).mean(0) if len(v[1]) else torch.zeros_like(dvec)
            vis.append(torch.cat([dvec, pvec]))
        x = self.visit_proj(torch.stack(vis))
        n = x.shape[0]
        idx = torch.clamp(torch.arange(n, device=dev), max=self.max_visits - 1)
        x = (x + self.pos(idx)).unsqueeze(0)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), 1)
        return self.encoder(x, mask=mask).squeeze(0)        # (n_visits, dim)

    def forward(self, patient, return_attn=False):
        """Logits for every visit (visit t only sees visits 0..t and meds of 0..t-1)."""
        h = self.encode(patient)
        s = self.state(h)
        logits = s @ self.drug_table().t() + self.bias       # (n, n_med)
        n = h.shape[0]
        attn = None
        if n > 1:
            dev = h.device
            meds = torch.zeros(n, self.n_med, device=dev)
            for i, v in enumerate(patient):
                meds[i, v[2]] = 1
            q, k = self.hist_q(h), self.hist_k(h)
            score = (q @ k.t()) / np.sqrt(h.shape[1])
            past = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev), -1)
            score = score.masked_fill(~past, float("-inf"))
            attn = torch.softmax(score[1:], -1)                # visit 0 has no history
            hist = torch.zeros_like(logits)
            hist[1:] = attn @ meds
            logits = logits + self.hist_w * hist
        if return_attn:
            return logits, attn
        return logits

    def ddi_loss(self, logits):
        p = torch.sigmoid(logits)                              # (n, n_med)
        return torch.einsum("ni,ij,nj->", p, self.ddi_adj, p) * 0.0005 / p.shape[0]

    @torch.no_grad()
    def forward_all(self, patient):
        return torch.sigmoid(self.forward(patient)).cpu().numpy()
