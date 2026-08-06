#!/usr/bin/env python
"""Build a LINEAGE-AWARE train/test split for PATRIC (reviewer #1).

Genomic distance between isolates is estimated from whole-genome UNITIG
presence/absence (pyseer format, produced by the AMR-GNN preprocessing), so the
split controls for clonal / lineage similarity rather than only repeated isolates.
Whole genomic clusters are assigned as a unit; the most genomically DISTINCT
clusters are placed in the test set so that test lineages are far from training.

Steps: (1) stream+subsample the unitig table -> isolate x unitig binary matrix;
(2) Jaccard distance + average-linkage clustering; (3) greedily assign the clusters
most distant from the rest to test (~test_frac of isolates); (4) filter the PATRIC
records into train/test manifests (same schema as splits_random).

Output: <out_dir>/split_patric_lineage_{train,test}.parquet  + a summary.
"""
import argparse, gzip, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform


def load_unitig_matrix(pyseer_gz, isolates, keep_every, max_unitigs):
    """Stream the pyseer unitig table; keep 1-in-keep_every informative lines until
    max_unitigs collected. Returns binary matrix [n_isolates, n_unitigs]."""
    iso_index = {s: i for i, s in enumerate(isolates)}
    n = len(isolates)
    lo, hi = max(2, int(0.02 * n)), int(0.98 * n)   # keep variable (accessory) unitigs only
    cols = []
    op = gzip.open if str(pyseer_gz).endswith(".gz") else open
    kept = 0
    with op(pyseer_gz, "rt") as fh:
        for ln, line in enumerate(fh):
            if ln % keep_every != 0:
                continue
            if " | " not in line:            # pyseer format: "UNITIG | sample1:1 sample2:1 ..."
                continue
            samps = line.rstrip("\n").split(" | ", 1)[1]
            vec = np.zeros(n, dtype=np.uint8)
            hit = 0
            for tok in samps.split():
                j = iso_index.get(tok.split(":")[0])
                if j is not None:
                    vec[j] = 1; hit += 1
            if lo <= hit <= hi:              # informative (intermediate-frequency) unitig
                cols.append(vec); kept += 1
                if kept >= max_unitigs:
                    break
    if not cols:
        raise RuntimeError("No informative unitigs collected; check ID matching.")
    return np.stack(cols, axis=1)  # [n_iso, n_unitig]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patric_manifest", required=True,
                    help="a manifest containing all PATRIC records (sample_id column)")
    ap.add_argument("--unitig", required=True, help="pyseer unitig presence/absence table (.gz) from the AMR-GNN preprocessing")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--mode", choices=["balanced", "divergent"], default="balanced",
                    help="balanced (default) = lineage-aware but train/test resistance profiles matched; "
                         "divergent = maximally far-homology test (extreme)")
    ap.add_argument("--n_tries", type=int, default=400, help="random whole-cluster partitions to search (balanced mode)")
    ap.add_argument("--n_clusters", type=int, default=40)
    ap.add_argument("--keep_every", type=int, default=37, help="subsample: keep 1 in N unitig lines")
    ap.add_argument("--max_unitigs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.patric_manifest)
    isolates = sorted(df["sample_id"].astype(str).unique().tolist())
    print(f"[lineage] {len(df)} records, {len(isolates)} unique isolates", flush=True)

    cache_npz = out / "unitig_matrix.npz"
    M = None
    if cache_npz.exists():
        z = np.load(cache_npz, allow_pickle=True)
        if list(z["isolates"]) == isolates:
            M = z["M"]; print(f"[lineage] loaded cached unitig matrix {M.shape}", flush=True)
    if M is None:
        print(f"[lineage] streaming unitigs (keep 1/{args.keep_every}, max {args.max_unitigs})...", flush=True)
        M = load_unitig_matrix(args.unitig, isolates, args.keep_every, args.max_unitigs)
        np.savez_compressed(cache_npz, M=M, isolates=np.array(isolates, dtype=object))
    print(f"[lineage] unitig matrix = {M.shape}; mean unitigs/iso = {M.sum(1).mean():.0f}", flush=True)

    # Jaccard distance between isolates
    D = pdist(M, metric="jaccard")            # condensed
    Dsq = squareform(D)
    Z = linkage(D, method="average")
    labels = fcluster(Z, t=args.n_clusters, criterion="maxclust")
    n_clu = len(set(labels))
    print(f"[lineage] {n_clu} genomic clusters", flush=True)

    # ---- assign WHOLE clusters to train/test (lineage-aware; related isolates never split) ----
    clu_ids = sorted(set(labels))
    clu_members = {c: np.where(labels == c)[0] for c in clu_ids}
    target = int(round(args.test_frac * len(isolates)))
    drugs = sorted(df["drug_name"].astype(str).unique())
    sid = df["sample_id"].astype(str)

    def score_partition(test_iso_set):
        te = df[sid.isin(test_iso_set)]; tr = df[~sid.isin(test_iso_set)]
        diffs = []
        for d in drugs:
            a = tr[tr.drug_name == d]["label"]; b = te[te.drug_name == d]["label"]
            if len(a) and len(b): diffs.append(abs(a.mean() - b.mean()))
        pd_diff = float(np.mean(diffs)) if diffs else 1.0
        r_diff = abs(tr["label"].mean() - te["label"].mean())
        size_pen = abs(len(test_iso_set) - target) / max(target, 1)
        return pd_diff + 0.5 * r_diff + 0.3 * size_pen, pd_diff, r_diff

    if args.mode == "divergent":
        def mean_dist_to_others(members):
            others = np.setdiff1d(np.arange(len(isolates)), members)
            return 0.0 if len(others) == 0 else float(Dsq[np.ix_(members, others)].mean())
        ranked = sorted(clu_ids, key=lambda c: -mean_dist_to_others(clu_members[c]))
        chosen, cnt = [], 0
        for c in ranked:
            if cnt >= target: break
            chosen.append(c); cnt += len(clu_members[c])
        test_isolates = {isolates[i] for c in chosen for i in clu_members[c]}
        _, pd_diff, r_diff = score_partition(test_isolates)
    else:  # balanced: many random whole-cluster partitions, keep the most composition-matched
        rng = np.random.default_rng(args.seed)
        best = None
        for _ in range(args.n_tries):
            order = list(clu_ids); rng.shuffle(order)
            chosen, cnt = [], 0
            for c in order:
                if cnt >= target: break
                chosen.append(c); cnt += len(clu_members[c])
            tis = {isolates[i] for c in chosen for i in clu_members[c]}
            sc, pdd, rdd = score_partition(tis)
            if best is None or sc < best[0]:
                best = (sc, tis, chosen, pdd, rdd)
        _, test_isolates, chosen, pd_diff, r_diff = best
    train_isolates = set(isolates) - test_isolates
    print(f"[lineage] mode={args.mode}: test={len(test_isolates)} iso in {len(chosen)} clusters | "
          f"per-drug R absdiff={pd_diff:.3f} overall R absdiff={r_diff:.3f}", flush=True)

    # cross train/test genomic separation (min distance between any train & test isolate)
    tr_idx = np.array([i for i in range(len(isolates)) if isolates[i] in train_isolates])
    te_idx = np.array([i for i in range(len(isolates)) if isolates[i] in test_isolates])
    cross = Dsq[np.ix_(te_idx, tr_idx)]
    sep = dict(min_cross_jaccard=float(cross.min()), mean_cross_jaccard=float(cross.mean()),
               mean_within_train=float(squareform(pdist(M[tr_idx], "jaccard")).mean()) if len(tr_idx) > 1 else None)

    tr_df = df[df["sample_id"].astype(str).isin(train_isolates)].copy()
    te_df = df[df["sample_id"].astype(str).isin(test_isolates)].copy()
    tr_df.to_parquet(out / "split_patric_lineage_train.parquet")
    te_df.to_parquet(out / "split_patric_lineage_test.parquet")

    summary = dict(
        mode=args.mode, n_isolates=len(isolates), n_clusters=n_clu, test_clusters=len(chosen),
        train_isolates=len(train_isolates), test_isolates=len(test_isolates),
        train_records=len(tr_df), test_records=len(te_df),
        train_R=float(tr_df["label"].mean()), test_R=float(te_df["label"].mean()),
        perdrug_R_absdiff=pd_diff, overall_R_absdiff=r_diff,
        genomic_separation=sep, unitig_matrix=list(M.shape))
    json.dump(summary, open(out / "lineage_split_summary.json", "w"), indent=1)
    print("[lineage] SUMMARY:", json.dumps(summary, indent=1), flush=True)
    print(f"[lineage] wrote {out}/split_patric_lineage_{{train,test}}.parquet", flush=True)


if __name__ == "__main__":
    main()
