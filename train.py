#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import tqdm
import joblib
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as PYGBatch
from torch_scatter import scatter_add


# -----------------------------
# Utils & Tokenizer
# -----------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_read_manifest(path: str) -> pd.DataFrame:
    p = str(path)
    if p.lower().endswith(".parquet"):
        return pd.read_parquet(p)
    if p.lower().endswith(".csv"):
        return pd.read_csv(p)
    raise ValueError(f"Unsupported manifest format: {p}")


class SmilesTokenizer:
    def __init__(self, max_len: int = 150):
        self.max_len = max_len
        self.pad_id = 0
        self.unk_id = 1
        self.stoi = {"<PAD>": self.pad_id, "<UNK>": self.unk_id}
        self.itos = ["<PAD>", "<UNK>"]

    def build_vocab(self, smiles_list: List[str], min_freq: int = 1):
        from collections import Counter
        c = Counter()
        for s in smiles_list:
            for ch in str(s):
                c[ch] += 1
        for ch, freq in sorted(c.items(), key=lambda x: (-x[1], x[0])):
            if freq >= min_freq and ch not in self.stoi:
                self.stoi[ch] = len(self.itos)
                self.itos.append(ch)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, smiles: str) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        s = str(smiles)
        ids = [self.stoi.get(ch, self.unk_id) for ch in s[: self.max_len]]
        if len(ids) < self.max_len:
            ids.extend([self.pad_id] * (self.max_len - len(ids)))
        ids_t = torch.tensor(ids, dtype=torch.long)
        key_padding_mask = (ids_t == self.pad_id)
        return ids_t, key_padding_mask


# -----------------------------
# Models (GVP + Transformer)
# -----------------------------
class DrugTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model=128, nhead=4, num_layers=3, max_len=150, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                               dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x, key_padding_mask=None):
        B, L = x.size()
        h = self.embedding(x) + self.pos_encoder[:, :L, :]
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).unsqueeze(-1).float()
            h_sum = torch.sum(h * valid, dim=1)
            denom = torch.clamp(valid.sum(dim=1), min=1e-9)
            out = h_sum / denom
        else:
            out = h.mean(dim=1)
        return self.fc_out(out)


class ProteinGVPEncoder(nn.Module):
    def __init__(self, node_in_dim=(6, 3), node_h_dim=(100, 16), edge_in_dim=(32, 1), edge_h_dim=(32, 1), num_layers=3,
                 drop_rate=0.1):
        super().__init__()
        import gvp
        self.W_v = nn.Sequential(gvp.GVP(node_in_dim, node_h_dim, activations=(None, None)), gvp.LayerNorm(node_h_dim))
        self.W_e = nn.Sequential(gvp.GVP(edge_in_dim, edge_h_dim, activations=(None, None)), gvp.LayerNorm(edge_h_dim))
        self.layers = nn.ModuleList(
            [gvp.GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) for _ in range(num_layers)])
        self.readout = nn.Sequential(nn.Linear(node_h_dim[0], 128), nn.ReLU())

    def forward(self, batch_data):
        h_V = self.W_v((batch_data.node_s, batch_data.node_v))
        h_E = self.W_e((batch_data.edge_s, batch_data.edge_v))
        edge_index = batch_data.edge_index
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out_scalar, _ = h_V
        batch = batch_data.batch
        num = scatter_add(out_scalar, batch, dim=0)
        den = scatter_add(torch.ones_like(out_scalar[:, 0]), batch, dim=0).unsqueeze(-1)
        return self.readout(num / (den + 1e-9))


class AMRPredictor(nn.Module):
    def __init__(self, vocab_size: int, n_genes: int):
        super().__init__()
        self.n_genes = n_genes
        self.drug_encoder = DrugTransformer(vocab_size=vocab_size)
        self.protein_encoder = ProteinGVPEncoder()
        self.fusion = nn.MultiheadAttention(embed_dim=128, num_heads=4, kdim=129, vdim=129, batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, drug_ids, drug_kpm, protein_batch, bg_index, present_mask_BG):
        device = drug_ids.device
        B = drug_ids.size(0)
        G = self.n_genes
        drug_emb = self.drug_encoder(drug_ids, drug_kpm)

        if protein_batch is not None and bg_index is not None and bg_index.numel() > 0:
            prot_flat = self.protein_encoder(protein_batch)
            prot_pack_flat = torch.zeros(B * G, 128, device=device, dtype=drug_emb.dtype)
            flat_idx = bg_index[:, 0] * G + bg_index[:, 1]
            prot_pack_flat.index_add_(0, flat_idx, prot_flat.to(dtype=prot_pack_flat.dtype))
            prot_pack = prot_pack_flat.view(B, G, 128)
        else:
            prot_pack = torch.zeros(B, G, 128, device=device, dtype=drug_emb.dtype)

        pm = present_mask_BG.to(device=device, dtype=torch.float32).unsqueeze(-1)
        prot_kv = torch.cat([prot_pack, pm], dim=-1)

        query = drug_emb.unsqueeze(1)
        attn_out, _ = self.fusion(query, prot_kv, prot_kv)
        x = self.norm(query + attn_out).squeeze(1)
        return self.head(x), None


# -----------------------------
# Smart Memory Dataset (核心逻辑)
# -----------------------------
class SmartMemoryDataset(Dataset):
    # 类变量：所有 Dataset 实例共享同一个图数据库，避免 train/val 重复占用内存
    _graph_bank = {}

    def __init__(self, manifest_df: pd.DataFrame, cache_root: str,
                 sample_col: str = "sample_id",
                 load_to_ram: bool = True,
                 n_jobs: int = 32):

        self.df = manifest_df.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.sample_col = sample_col
        self.load_to_ram = load_to_ram

        # 确保 sample_id 都是字符串，防止 int/str 混用导致匹配失败
        self.df[self.sample_col] = self.df[self.sample_col].astype(str)

        if self.load_to_ram:
            self._preload_graphs(n_jobs)

    def _preload_graphs(self, n_jobs):
        """
        只加载 Unique 的 sample_id 对应的图结构。
        """
        # 1. 找出当前数据集需要哪些 sample_id
        needed_ids = self.df[self.sample_col].unique()

        # 2. 找出哪些还没加载进内存 (_graph_bank)
        missing_ids = [sid for sid in needed_ids if sid not in SmartMemoryDataset._graph_bank]

        if len(missing_ids) == 0:
            return  # 已经都有了

        print(f"Loading {len(missing_ids)} new unique graph structures into RAM...")

        # 3. 只需要找到任意一个包含该 sample_id 的文件即可
        # 我们创建一个 map: sample_id -> pt_path
        # drop_duplicates 保留第一条即可
        subset = self.df[self.df[self.sample_col].isin(missing_ids)]
        unique_map = subset.drop_duplicates(subset=[self.sample_col])
        tasks = list(zip(unique_map[self.sample_col], unique_map["pt_path"]))

        # 4. 并行读取
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self._load_one_graph)(sid, path, self.cache_root)
            for sid, path in tqdm.tqdm(tasks, desc="Smart Caching")
        )

        # 5. 存入 Bank
        for res in results:
            if res:
                sid, data = res
                SmartMemoryDataset._graph_bank[sid] = data

        print(f"Global Graph Bank Size: {len(SmartMemoryDataset._graph_bank)} unique isolates.")

    @staticmethod
    def _load_one_graph(sid, pt_path, root):
        p = Path(pt_path)
        if not p.is_absolute():
            p = root / p
        try:
            # 加载整个 .pt
            obj = torch.load(p, map_location="cpu")
            # 只提取图结构部分，扔掉 drug info 和 label
            return sid, {
                "graphs_present": obj.get("graphs_present", []),
                "gene_indices": obj.get("gene_indices", torch.zeros(0, dtype=torch.long)),
                "present_mask": obj.get("present_mask")
            }
        except Exception as e:
            print(f"Failed to load {p}: {e}")
            return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        sid = row[self.sample_col]

        # 1. 直接从 manifest 获取 SMILES 和 Label (极快)
        smiles = str(row["drug_smiles"])
        label = float(row["label"])

        # 2. 从内存 Bank 获取图结构
        if self.load_to_ram and sid in SmartMemoryDataset._graph_bank:
            g_data = SmartMemoryDataset._graph_bank[sid]
        else:
            # Fallback (如果没开 RAM 模式，还是得读文件，虽然低效但兼容)
            _, g_data = self._load_one_graph(sid, row["pt_path"], self.cache_root)

        return {
            "drug_smiles": smiles,
            "label": label,
            "graphs_present": g_data["graphs_present"],
            "gene_indices": g_data["gene_indices"],
            "present_mask": g_data["present_mask"]
        }


def collate_smart(batch, tokenizer):
    drug_ids, drug_kpm, labels, masks = [], [], [], []
    all_graphs, bg_pairs = [], []

    for bidx, rec in enumerate(batch):
        ids, kpm = tokenizer.encode(rec["drug_smiles"])
        drug_ids.append(ids)
        drug_kpm.append(kpm)
        labels.append(rec["label"])

        pm = rec["present_mask"]
        masks.append(pm if pm is not None else torch.zeros(1))

        g_idxs = rec["gene_indices"]
        graphs = rec["graphs_present"]

        if g_idxs is not None and graphs is not None:
            if isinstance(g_idxs, list): g_idxs = torch.tensor(g_idxs)
            all_graphs.extend(graphs)
            for gi in g_idxs.tolist():
                bg_pairs.append((bidx, gi))

    drug_ids = torch.stack(drug_ids, 0)
    drug_kpm = torch.stack(drug_kpm, 0)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    present_mask_BG = torch.stack(masks, 0)

    if len(all_graphs) > 0:
        protein_batch = PYGBatch.from_data_list(all_graphs)
        bg_index = torch.tensor(bg_pairs, dtype=torch.long)
    else:
        protein_batch, bg_index = None, None

    return drug_ids, drug_kpm, protein_batch, bg_index, present_mask_BG, y


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for drug_ids, drug_kpm, prot_batch, bg_index, pm_BG, y in loader:
        drug_ids, drug_kpm = drug_ids.to(device, non_blocking=True), drug_kpm.to(device, non_blocking=True)
        pm_BG, y = pm_BG.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if prot_batch: prot_batch = prot_batch.to(device)
        if bg_index is not None: bg_index = bg_index.to(device, non_blocking=True)

        logits, _ = model(drug_ids, drug_kpm, prot_batch, bg_index, pm_BG)
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--test_manifest", default=None)
    ap.add_argument("--n_genes", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    # 你的 sample_id 列名，默认就是 sample_id
    ap.add_argument("--sample_col", default="sample_id", help="Column name for unique isolate ID")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--load_to_ram", action="store_true")
    ap.add_argument("--cache_jobs", type=int, default=32)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = str(Path(args.cache_root))

    print(f"Reading Manifests...")
    train_df = safe_read_manifest(args.train_manifest)
    val_df = safe_read_manifest(args.val_manifest)
    test_df = safe_read_manifest(args.test_manifest) if args.test_manifest else None

    # Tokenizer
    tokenizer = SmilesTokenizer()
    tokenizer.build_vocab(train_df["drug_smiles"].astype(str).tolist())

    # Datasets
    # 初始化 Dataset 时会自动触发 Smart Caching
    # Train, Val, Test 会自动检测并只加载新的 isolates
    print("--- Initializing Train Dataset ---")
    train_ds = SmartMemoryDataset(train_df, cache_root, args.sample_col, args.load_to_ram, args.cache_jobs)

    print("--- Initializing Val Dataset ---")
    val_ds = SmartMemoryDataset(val_df, cache_root, args.sample_col, args.load_to_ram, args.cache_jobs)

    test_ds = None
    if test_df is not None:
        print("--- Initializing Test Dataset ---")
        test_ds = SmartMemoryDataset(test_df, cache_root, args.sample_col, args.load_to_ram, args.cache_jobs)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": lambda b: collate_smart(b, tokenizer),
        "persistent_workers": args.num_workers > 0,
        "prefetch_factor": 2 if args.num_workers > 0 else None
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs) if test_ds else None

    model = AMRPredictor(tokenizer.vocab_size, args.n_genes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = -1.0

    print(f"Training Start. Steps per epoch: {len(train_loader)}")

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        opt.zero_grad(set_to_none=True)

        pbar = tqdm.tqdm(train_loader, desc=f"Ep {ep}", leave=False)
        for step, (drug_ids, drug_kpm, prot_batch, bg_index, pm_BG, y) in enumerate(pbar):
            drug_ids, drug_kpm = drug_ids.to(device, non_blocking=True), drug_kpm.to(device, non_blocking=True)
            pm_BG, y = pm_BG.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if prot_batch: prot_batch = prot_batch.to(device)
            if bg_index is not None: bg_index = bg_index.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits, _ = model(drug_ids, drug_kpm, prot_batch, bg_index, pm_BG)
                loss = loss_fn(logits, y) / args.grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * args.grad_accum_steps

            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).float()
                correct += (pred == y).sum().item()
                total += y.numel()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1 == len(train_loader)):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=f"{loss.item() * args.grad_accum_steps:.4f}")

        train_acc = correct / max(total, 1)
        val_acc = eval_acc(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)

        print(f"[Epoch {ep}] Loss={avg_loss:.4f} TrainAcc={train_acc:.4f} ValAcc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")


if __name__ == "__main__":
    main()