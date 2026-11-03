"""GAMENet (Shang et al., AAAI 2019) ported to current PyTorch.

Original code: https://github.com/sjy1203/GAMENet/blob/master/code/models.py
The architecture and hyperparameters follow the original. Changes are only
API updates (torch.sigmoid, explicit softmax dims, device handling) so it runs
on PyTorch 2.x, plus a forward_all() helper used at evaluation time.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        stdv = 1.0 / math.sqrt(out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        return adj @ (x @ self.weight) + self.bias


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj):
        super().__init__()
        adj = adj + np.eye(adj.shape[0])
        rowsum = adj.sum(1)
        r_inv = np.where(rowsum > 0, 1.0 / rowsum, 0.0)
        adj = np.diag(r_inv) @ adj
        self.register_buffer("adj", torch.tensor(adj, dtype=torch.float32))
        self.register_buffer("x", torch.eye(voc_size))
        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        h = F.relu(self.gcn1(self.x, self.adj))
        return self.gcn2(self.dropout(h), self.adj)


class GAMENet(nn.Module):
    def __init__(self, voc_size, ehr_adj, ddi_adj, emb_dim=64, ddi_in_memory=True):
        super().__init__()
        self.voc_size = voc_size
        self.ddi_in_memory = ddi_in_memory
        self.register_buffer("ddi_adj", torch.tensor(ddi_adj, dtype=torch.float32))
        self.embeddings = nn.ModuleList([nn.Embedding(voc_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(0.4)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim * 4, emb_dim))
        self.ehr_gcn = GCN(voc_size[2], emb_dim, ehr_adj)
        self.ddi_gcn = GCN(voc_size[2], emb_dim, ddi_adj)
        self.inter = nn.Parameter(torch.empty(1))
        self.output = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim * 3, emb_dim * 2),
                                    nn.ReLU(), nn.Linear(emb_dim * 2, voc_size[2]))
        for e in self.embeddings:
            e.weight.data.uniform_(-0.1, 0.1)
        self.inter.data.uniform_(-0.1, 0.1)

    def _queries(self, patient):
        dev = self.inter.device
        seqs = []
        for k in range(2):
            vis = [self.dropout(self.embeddings[k](torch.tensor(v[k], device=dev))).mean(0) for v in patient]
            out, _ = self.encoders[k](torch.stack(vis).unsqueeze(0))
            seqs.append(out.squeeze(0))
        return self.query(torch.cat(seqs, -1))  # (n_visits, dim)

    def _predict(self, queries, patient, t, drug_memory):
        """Prediction for visit t using visits 0..t (GAMENet reads history meds 0..t-1)."""
        q = queries[t:t + 1]
        fact1 = F.softmax(q @ drug_memory.t(), -1) @ drug_memory
        if t > 0:
            keys = queries[:t]
            values = torch.zeros(t, self.voc_size[2], device=q.device)
            for i in range(t):
                values[i, patient[i][2]] = 1
            fact2 = (F.softmax(q @ keys.t(), -1) @ values) @ drug_memory
        else:
            fact2 = fact1
        return self.output(torch.cat([q, fact1, fact2], -1))

    def memory(self):
        if self.ddi_in_memory:
            return self.ehr_gcn() - self.ddi_gcn() * self.inter
        return self.ehr_gcn()

    def forward(self, patient):
        """Logits for the LAST visit of `patient` (training mode, like the original)."""
        queries = self._queries(patient)
        logits = self._predict(queries, patient, len(patient) - 1, self.memory())
        prob = torch.sigmoid(logits)
        batch_neg = (prob.t() * prob).mul(self.ddi_adj).mean()
        return logits, batch_neg

    @torch.no_grad()
    def forward_all(self, patient):
        """Probabilities for every visit, each using only its own past."""
        queries = self._queries(patient)
        mem = self.memory()
        return torch.cat([torch.sigmoid(self._predict(queries, patient, t, mem))
                          for t in range(len(patient))]).cpu().numpy()
