#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from torch_geometric.data import Batch as PYGBatch


AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


# -----------------------------
# Tokenizer (compatible with ckpt)
# -----------------------------
class SmilesTokenizer:
    def __init__(self, max_len: int = 150):
        self.max_len = max_len
        self.pad_id = 0
        self.unk_id = 1
        self.stoi = {"<PAD>": self.pad_id, "<UNK>": self.unk_id}
        self.itos = ["<PAD>", "<UNK>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def load_from_ckpt(self, stoi: Dict[str, int], max_len: int):
        self.stoi = dict(stoi)
        self.itos = [None] * (max(self.stoi.values()) + 1)
        for k, v in self.stoi.items():
            self.itos[v] = k
        self.max_len = int(max_len)

    def encode(self, smiles: str) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        s = str(smiles)
        ids = [self.stoi.get(ch, self.unk_id) for ch in s[: self.max_len]]
        if len(ids) < self.max_len:
            ids.extend([self.pad_id] * (self.max_len - len(ids)))
        ids_t = torch.tensor(ids, dtype=torch.long)
        key_padding_mask = (ids_t == self.pad_id)
        return ids_t, key_padding_mask


# -----------------------------
# Drug Transformer with attention capture
#   (parameter shapes compatible with nn.TransformerEncoderLayer)
# -----------------------------
class CapturingTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        attn_out, attn_w = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        src = self.norm1(src + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src, attn_w  # [B,H,L,L]


class DrugTransformerWithAttn(nn.Module):
    def __init__(self, vocab_size: int, d_model=128, nhead=4, num_layers=3,
                 max_len=150, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model))
        self.layers = nn.ModuleList([
            CapturingTransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.LongTensor, key_padding_mask: Optional[torch.BoolTensor] = None):
        B, L = x.size()
        h = self.embedding(x) + self.pos_encoder[:, :L, :]
        attn_all: List[torch.Tensor] = []
        for layer in self.layers:
            h, attn_w = layer(h, src_key_padding_mask=key_padding_mask)
            attn_all.append(attn_w)

        if key_padding_mask is not None:
            valid = (~key_padding_mask).unsqueeze(-1).float()
            out = (h * valid).sum(dim=1) / torch.clamp(valid.sum(dim=1), min=1e-9)
        else:
            out = h.mean(dim=1)
        return self.fc_out(out), attn_all


# -----------------------------
# Protein encoder + cross-attn (same logic as training)
# -----------------------------
class ProteinGVPEncoder(nn.Module):
    def __init__(self,
                 node_in_dim=(6, 3),
                 node_h_dim=(100, 16),
                 edge_in_dim=(32, 1),
                 edge_h_dim=(32, 1),
                 num_layers: int = 3,
                 drop_rate: float = 0.1):
        super().__init__()
        import gvp
        from torch_scatter import scatter_add  # noqa: F401

        self.W_v = nn.Sequential(
            gvp.GVP(node_in_dim, node_h_dim, activations=(None, None)),
            gvp.LayerNorm(node_h_dim),
        )
        self.W_e = nn.Sequential(
            gvp.GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            gvp.LayerNorm(edge_h_dim),
        )
        self.layers = nn.ModuleList([
            gvp.GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(node_h_dim[0], 128),
            nn.ReLU(),
        )

    def forward(self, batch_data):
        from torch_scatter import scatter_add

        nodes = (batch_data.node_s, batch_data.node_v)
        edges = (batch_data.edge_s, batch_data.edge_v)
        edge_index = batch_data.edge_index

        h_V = self.W_v(nodes)
        h_E = self.W_e(edges)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)

        out_scalar, _ = h_V
        batch = batch_data.batch

        if hasattr(batch_data, "node_weight") and batch_data.node_weight is not None:
            w = torch.clamp(batch_data.node_weight, min=0.0).to(out_scalar.dtype)
            num = scatter_add(out_scalar * w.unsqueeze(-1), batch, dim=0)
            den = scatter_add(w, batch, dim=0).unsqueeze(-1)
            emb = num / (den + 1e-9)
            emb = emb.masked_fill((den.squeeze(-1) <= 0.0).unsqueeze(-1), 0.0)
        else:
            ones = torch.ones(out_scalar.size(0), device=out_scalar.device, dtype=out_scalar.dtype)
            num = scatter_add(out_scalar, batch, dim=0)
            den = scatter_add(ones, batch, dim=0).unsqueeze(-1)
            emb = num / (den + 1e-9)

        return self.readout(emb)


class DrugProteinAttention(nn.Module):
    def __init__(self, embed_dim_q=128, kv_dim=129, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim_q,
            num_heads=num_heads,
            kdim=kv_dim,
            vdim=kv_dim,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim_q)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim_q, embed_dim_q),
            nn.ReLU(),
            nn.Linear(embed_dim_q, embed_dim_q),
        )

    def forward(self, drug_q, prot_kv):
        out, w = self.attn(
            drug_q, prot_kv, prot_kv,
            need_weights=True,
            average_attn_weights=False
        )
        x = self.norm(drug_q + out)
        x = x + self.ffn(x)
        return x.squeeze(1), w  # [B, heads, 1, G]


class AMRPredictorExplainable(nn.Module):
    def __init__(self, vocab_size: int, n_genes: int):
        super().__init__()
        self.n_genes = n_genes
        self.drug_encoder = DrugTransformerWithAttn(vocab_size=vocab_size, max_len=150)
        self.protein_encoder = ProteinGVPEncoder()
        self.fusion = DrugProteinAttention(embed_dim_q=128, kv_dim=129, num_heads=4)
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self,
                drug_ids: torch.Tensor,
                drug_kpm: torch.Tensor,
                protein_batch: Optional[PYGBatch],
                bg_index: Optional[torch.LongTensor],
                present_mask_BG: torch.Tensor):
        device = drug_ids.device
        B = drug_ids.size(0)
        G = self.n_genes

        drug_emb, drug_self_attn = self.drug_encoder(drug_ids, drug_kpm)

        prot_pack_flat = torch.zeros(B * G, 128, device=device, dtype=drug_emb.dtype)
        if protein_batch is not None and bg_index is not None and bg_index.numel() > 0:
            prot_flat = self.protein_encoder(protein_batch)
            b_idx = bg_index[:, 0].to(device=device)
            g_idx = bg_index[:, 1].to(device=device)
            flat_idx = b_idx * G + g_idx
            prot_pack_flat.index_add_(0, flat_idx, prot_flat.to(dtype=prot_pack_flat.dtype))

        prot_pack = prot_pack_flat.view(B, G, 128)
        pm = present_mask_BG.to(device=device, dtype=torch.float32).unsqueeze(-1)
        prot_kv = torch.cat([prot_pack, pm], dim=-1)

        ctx, cross_attn = self.fusion(drug_emb.unsqueeze(1), prot_kv)
        logits = self.head(torch.cat([drug_emb, ctx], dim=-1))
        return logits, drug_self_attn, cross_attn


# -----------------------------
# IO helpers
# -----------------------------
def read_manifest(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("manifest must be .parquet or .csv")


def load_pt(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def make_batch_from_pts(pts: List[Dict[str, Any]], tokenizer: SmilesTokenizer):
    drug_ids_list, drug_kpm_list = [], []
    all_graphs_present: List[Any] = []
    bg_pairs: List[Tuple[int, int]] = []
    masks: List[torch.Tensor] = []

    for bidx, obj in enumerate(pts):
        ids, kpm = tokenizer.encode(obj["drug_smiles"])
        drug_ids_list.append(ids)
        drug_kpm_list.append(kpm)

        pm = obj["present_mask"]
        if isinstance(pm, list):
            pm = torch.tensor(pm, dtype=torch.float32)
        masks.append(pm)

        gene_indices = obj.get("gene_indices", torch.zeros(0, dtype=torch.long))
        graphs_present = obj.get("graphs_present", [])
        if isinstance(gene_indices, list):
            gene_indices = torch.tensor(gene_indices, dtype=torch.long)

        for gi, g in zip(gene_indices.tolist(), graphs_present):
            all_graphs_present.append(g)
            bg_pairs.append((bidx, gi))

    drug_ids = torch.stack(drug_ids_list, dim=0)
    drug_kpm = torch.stack(drug_kpm_list, dim=0)
    present_mask_BG = torch.stack(masks, dim=0)

    if len(all_graphs_present) == 0:
        protein_batch = None
        bg_index = None
    else:
        protein_batch = PYGBatch.from_data_list(all_graphs_present)
        bg_index = torch.tensor(bg_pairs, dtype=torch.long)

    return drug_ids, drug_kpm, protein_batch, bg_index, present_mask_BG


def save_heatmap(mat: np.ndarray, out_png: Path, title: str, tokens: Optional[List[str]] = None):
    plt.figure(figsize=(10, 8))
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    if tokens is not None and len(tokens) == mat.shape[0]:
        plt.xticks(np.arange(len(tokens)), tokens, rotation=90, fontsize=6)
        plt.yticks(np.arange(len(tokens)), tokens, fontsize=6)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def summarize_attention_incoming(attn_LxL: np.ndarray) -> np.ndarray:
    # incoming to position j
    return attn_LxL.sum(axis=0)


# -----------------------------
# Mapping: node_index -> PDB residue
#   consistent with ConstructAlgo and your training template parsing
# -----------------------------
def safe_three_to_one(resname: str) -> str:
    try:
        return seq1(resname, custom_map={"UNK": "X"})
    except Exception:
        return "X"


def build_node_map_csv(struct_pdb: Path, out_csv: Path) -> pd.DataFrame:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", str(struct_pdb))
    model = next(structure.get_models())

    rows = []
    node_index = 0
    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            aa = safe_three_to_one(res.get_resname())
            if aa not in AA_SET:
                continue
            hetflag, resseq, icode = res.get_id()
            icode = (icode or "").strip()
            rows.append({
                "node_index": node_index,
                "chain_id": chain.id,
                "pdb_resseq": int(resseq),
                "icode": icode,
                "resseq_with_icode": f"{resseq}{icode}" if icode else f"{resseq}",
                "resname3": res.get_resname(),
                "aa": aa,
            })
            node_index += 1

    if len(rows) == 0:
        raise RuntimeError(f"No standard residues extracted from: {struct_pdb}")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def load_or_build_node_map(gene_id: str, structure_dir: Path, map_dir: Path) -> Tuple[pd.DataFrame, Path]:
    map_path = map_dir / f"{gene_id}_node_to_pdb_residue.csv"
    if map_path.exists():
        df = pd.read_csv(map_path)
        return df, map_path

    pdb_path = structure_dir / f"{gene_id}.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(f"Missing PDB: {pdb_path}")
    df = build_node_map_csv(pdb_path, map_path)
    return df, map_path


def hotspot_mapping_csv(gene_id: str, hotspot_dir: Path, node_map: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    hot_path = hotspot_dir / f"{gene_id}_hotspots.npy"
    if not hot_path.exists():
        raise FileNotFoundError(f"Missing hotspots: {hot_path}")

    hot = np.load(hot_path).astype(int)
    hot = hot[(hot >= 0) & (hot < len(node_map))]
    hot = np.unique(hot)

    df_hot = node_map.iloc[hot].copy()
    df_hot.insert(0, "hotspot_rank", np.arange(len(df_hot), dtype=int))
    df_hot.insert(1, "node_index", hot.astype(int))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_hot.to_csv(out_csv, index=False)
    return df_hot


def write_pymol_script(out_pml: Path, structure_dir: Path, top_gene_ids: List[str], hotspot_csvs: Dict[str, Path]):
    """
    Generates a PyMOL script that:
      - loads each top protein PDB
      - selects hotspot residues using chain+resseq+icode info from hotspot CSV
      - shows cartoon + sticks for hotspots
    """
    lines = []
    lines.append("reinitialize\n")
    lines.append("bg_color white\n")
    lines.append("set cartoon_fancy_helices, 1\n")
    lines.append("set cartoon_transparency, 0.1\n")
    lines.append("set stick_radius, 0.2\n")
    lines.append("set ray_opaque_background, off\n\n")

    for i, gene_id in enumerate(top_gene_ids, start=1):
        pdb_path = structure_dir / f"{gene_id}.pdb"
        obj_name = f"prot{i}_{gene_id}"
        lines.append(f"load {pdb_path.as_posix()}, {obj_name}\n")
        lines.append(f"hide everything, {obj_name}\n")
        lines.append(f"show cartoon, {obj_name}\n")

        # Build selection string from hotspot CSV: (chain A and resi 12+icode)
        csv_path = hotspot_csvs[gene_id]
        df = pd.read_csv(csv_path)
        sels = []
        for _, r in df.iterrows():
            chain = str(r["chain_id"])
            resi = str(r["pdb_resseq"])
            icode = str(r.get("icode", "")).strip()
            # PyMOL insertion code uses resi like "12A" if needed
            resi_full = f"{resi}{icode}" if icode else resi
            sels.append(f"(chain {chain} and resi {resi_full})")

        if sels:
            sel_expr = " or ".join(sels)
            sel_name = f"hot_{obj_name}"
            lines.append(f"select {sel_name}, {obj_name} and ({sel_expr})\n")
            lines.append(f"show sticks, {sel_name}\n")
            lines.append(f"color red, {sel_name}\n")
            lines.append(f"set stick_transparency, 0.0, {sel_name}\n")
        lines.append("\n")

    lines.append("# Optional: render\n")
    lines.append("# ray 2000,1500\n")
    lines.append("# png hotspots.png, dpi=300\n")

    out_pml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pml, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser(description="End-to-end interpretability pipeline for a given drug_name.")

    ap.add_argument("--ckpt", required=True, help="best.pt / ckpt_epoch*.pt")
    ap.add_argument("--n_genes", type=int, required=True, help="e.g., 333")

    ap.add_argument("--manifest", required=True, help="manifest (.parquet/.csv), must contain pt_path, drug_name")
    ap.add_argument("--cache_root", required=True, help="root directory for relative pt_path")

    ap.add_argument("--panel_map_csv", required=True, help="CSV: gene_index,gene_id (panel order)")

    ap.add_argument("--drug_name", required=True)
    ap.add_argument("--max_samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--layer", type=int, default=-1, help="drug self-attn layer (-1 last)")
    ap.add_argument("--head", type=int, default=-1, help="drug self-attn head (-1 mean over heads)")
    ap.add_argument("--top_proteins", type=int, default=10)

    ap.add_argument("--structure_dir", required=True, help="directory with {gene_id}.pdb")
    ap.add_argument("--hotspot_dir", required=True, help="directory with {gene_id}_hotspots.npy")
    ap.add_argument("--map_dir", required=True, help="output dir to store node maps per gene")

    ap.add_argument("--out_dir", required=True)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    structure_dir = Path(args.structure_dir)
    hotspot_dir = Path(args.hotspot_dir)
    map_dir = Path(args.map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load ckpt + model
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    tok = SmilesTokenizer(max_len=int(ckpt["tokenizer"]["max_len"]))
    tok.load_from_ckpt(ckpt["tokenizer"]["stoi"], ckpt["tokenizer"]["max_len"])

    model = AMRPredictorExplainable(vocab_size=tok.vocab_size, n_genes=args.n_genes).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # -----------------------------
    # Load manifest and sample pt files for this drug
    # -----------------------------
    df = read_manifest(args.manifest)
    if "pt_path" not in df.columns or "drug_name" not in df.columns:
        raise ValueError("manifest must contain columns: pt_path, drug_name")

    sub = df[df["drug_name"].astype(str) == str(args.drug_name)].reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(f"No rows found for drug_name={args.drug_name}")

    sub = sub.sample(n=min(args.max_samples, len(sub)), random_state=args.seed).reset_index(drop=True)

    cache_root = Path(args.cache_root)
    pts: List[Dict[str, Any]] = []
    used_paths: List[str] = []
    for p in sub["pt_path"].astype(str).tolist():
        pp = Path(p)
        if not pp.is_absolute():
            pp = cache_root / pp
        pts.append(load_pt(str(pp)))
        used_paths.append(str(pp))

    # -----------------------------
    # Run model once to capture attentions
    # -----------------------------
    drug_ids, drug_kpm, prot_batch, bg_index, pm_BG = make_batch_from_pts(pts, tok)
    drug_ids = drug_ids.to(device)
    drug_kpm = drug_kpm.to(device)
    pm_BG = pm_BG.to(device)
    if prot_batch is not None:
        prot_batch = prot_batch.to(device)
    if bg_index is not None:
        bg_index = bg_index.to(device)

    with torch.no_grad():
        _, drug_self_attn, cross_attn = model(drug_ids, drug_kpm, prot_batch, bg_index, pm_BG)

    # -----------------------------
    # Drug self-attention heatmap + token importance
    # -----------------------------
    n_layers = len(drug_self_attn)
    layer_idx = args.layer if args.layer >= 0 else (n_layers + args.layer)
    layer_attn = drug_self_attn[layer_idx].detach().cpu().numpy()  # [B,H,L,L]

    if args.head >= 0:
        mat = layer_attn[:, args.head, :, :].mean(axis=0)
        title = f"Drug self-attn | drug={args.drug_name} | layer={layer_idx} | head={args.head}"
    else:
        mat = layer_attn.mean(axis=(0, 1))
        title = f"Drug self-attn | drug={args.drug_name} | layer={layer_idx} | mean heads"

    rep_smiles = str(pts[0]["drug_smiles"])
    toks = list(rep_smiles[:tok.max_len])
    L = mat.shape[0]
    if len(toks) < L:
        toks = toks + ["<PAD>"] * (L - len(toks))
    toks = toks[:L]

    heat_png = out_dir / f"drug_self_attn__{args.drug_name}__layer{layer_idx}__head{args.head}.png"
    save_heatmap(mat, heat_png, title, tokens=toks)

    token_scores = summarize_attention_incoming(mat)
    token_df = pd.DataFrame({
        "pos": np.arange(L, dtype=int),
        "token": toks,
        "incoming_attention_sum": token_scores.astype(float),
    }).sort_values("incoming_attention_sum", ascending=False)
    token_csv = out_dir / f"drug_token_importance__{args.drug_name}.csv"
    token_df.to_csv(token_csv, index=False)

    # -----------------------------
    # Cross-attention protein ranking
    # -----------------------------
    # cross_attn: [B, heads, 1, G] -> [G]
    ca = cross_attn.detach().cpu().numpy().squeeze(2)  # [B, heads, G]
    prot_scores = ca.mean(axis=(0, 1))  # [G]
    rank = np.argsort(-prot_scores)

    panel_map = pd.read_csv(args.panel_map_csv)
    if "gene_index" not in panel_map.columns or "gene_id" not in panel_map.columns:
        raise ValueError("panel_map_csv must contain columns: gene_index, gene_id")
    panel_map = panel_map.copy()
    panel_map["gene_index"] = panel_map["gene_index"].astype(int)
    geneid_by_idx = dict(zip(panel_map["gene_index"].tolist(), panel_map["gene_id"].astype(str).tolist()))

    top_gene_ids: List[str] = []
    rows = []
    for gi in rank[: args.top_proteins]:
        gene_id = geneid_by_idx.get(int(gi), f"IDX_{int(gi)}")
        top_gene_ids.append(gene_id)
        rows.append({
            "rank": int(len(rows) + 1),
            "gene_index": int(gi),
            "gene_id": gene_id,
            "cross_attn_score": float(prot_scores[int(gi)]),
        })
    prot_df = pd.DataFrame(rows)
    prot_csv = out_dir / f"protein_importance__{args.drug_name}.csv"
    prot_df.to_csv(prot_csv, index=False)

    # -----------------------------
    # For each top protein: ensure node map exists + export hotspot mapping
    # -----------------------------
    hotspot_csvs: Dict[str, Path] = {}
    hotspot_out_dir = out_dir / "hotspots"
    hotspot_out_dir.mkdir(parents=True, exist_ok=True)

    for gene_id in top_gene_ids:
        # gene_id might be IDX_* if mapping missing
        if gene_id.startswith("IDX_"):
            continue

        node_map_df, map_path = load_or_build_node_map(gene_id, structure_dir, map_dir)

        out_hot_csv = hotspot_out_dir / f"{gene_id}_hotspot_residues.csv"
        try:
            _ = hotspot_mapping_csv(gene_id, hotspot_dir, node_map_df, out_hot_csv)
            hotspot_csvs[gene_id] = out_hot_csv
        except FileNotFoundError as e:
            # hotspots may be missing for some genes; continue
            miss = hotspot_out_dir / f"{gene_id}_MISSING_HOTSPOTS.txt"
            with open(miss, "w", encoding="utf-8") as f:
                f.write(str(e) + "\n")

    # -----------------------------
    # Write PyMOL script to visualize hotspots for top proteins
    # -----------------------------
    out_pml = out_dir / f"view_hotspots__{args.drug_name}.pml"
    # Only include genes that have hotspot CSV
    pymol_gene_ids = [g for g in top_gene_ids if g in hotspot_csvs]
    if len(pymol_gene_ids) > 0:
        write_pymol_script(out_pml, structure_dir, pymol_gene_ids, hotspot_csvs)

    # -----------------------------
    # Write a concise report
    # -----------------------------
    rpt = out_dir / f"REPORT__{args.drug_name}.txt"
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"Drug interpretability pipeline report\n")
        f.write(f"Drug: {args.drug_name}\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Samples used: {len(pts)} (seed={args.seed})\n\n")

        f.write("Outputs:\n")
        f.write(f"  Drug self-attn heatmap: {heat_png}\n")
        f.write(f"  Drug token importance: {token_csv}\n")
        f.write(f"  Protein importance: {prot_csv}\n")
        if len(pymol_gene_ids) > 0:
            f.write(f"  PyMOL hotspots script: {out_pml}\n")
        f.write("\nTop proteins:\n")
        for _, r in prot_df.iterrows():
            f.write(f"  rank={int(r['rank']):02d} gene_index={int(r['gene_index']):03d} "
                    f"gene_id={r['gene_id']} score={float(r['cross_attn_score']):.6f}\n")

        f.write("\nHotspot mapping CSVs generated:\n")
        for gene_id, csvp in hotspot_csvs.items():
            f.write(f"  {gene_id}: {csvp}\n")

        f.write("\nExample pt paths used (first 10):\n")
        for p in used_paths[:10]:
            f.write(f"  {p}\n")

    print(f"[Done] Report dir: {out_dir}")
    print(f"  - {rpt}")
    print(f"  - {heat_png}")
    print(f"  - {prot_csv}")
    if len(pymol_gene_ids) > 0:
        print(f"  - {out_pml}")
    if len(hotspot_csvs) > 0:
        print(f"  - hotspots/: {len(hotspot_csvs)} csv files")


if __name__ == "__main__":
    main()
