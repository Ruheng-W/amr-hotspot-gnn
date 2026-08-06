#!/usr/bin/env python
"""Build a LEAVE-SPECIES-OUT (cross-species) train/test split for the multi-species
Antibiogram cohort (supplement to Exp-1 / reviewer #1: species identity is highly
predictive in a multispecies dataset). Whole bacterial species are held out for test,
so the test species are never seen during training. Records for a species go entirely
to one side.

Output: <out_dir>/split_antibiogram_species_{train,test}.parquet + summary.
"""
import argparse, json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--antibiogram_manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--hold_species", nargs="*", default=None,
                    help="explicit species names to hold out; if unset, auto-select to ~test_frac")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.antibiogram_manifest)
    assert "species_name" in df.columns, "manifest has no species_name column"
    per_sp = df.groupby("species_name").agg(records=("label", "size"),
                                            isolates=("sample_id", "nunique"),
                                            R=("label", "mean")).reset_index()
    per_sp = per_sp.sort_values("records", ascending=False)
    print("[species] per-species counts:\n", per_sp.to_string(index=False), flush=True)

    total = len(df)
    if args.hold_species:
        held = list(args.hold_species)
    else:
        # greedily hold out mid/small species (ascending size) until ~test_frac of records,
        # so the largest species stay in TRAIN (keeps training diverse).
        target = args.test_frac * total
        held, acc = [], 0
        for _, row in per_sp.sort_values("records").iterrows():
            if acc >= target:
                break
            held.append(row["species_name"]); acc += row["records"]
    held_set = set(held)
    te_df = df[df["species_name"].isin(held_set)].copy()
    tr_df = df[~df["species_name"].isin(held_set)].copy()
    tr_df.to_parquet(out / "split_antibiogram_species_train.parquet")
    te_df.to_parquet(out / "split_antibiogram_species_test.parquet")

    summary = dict(
        held_out_species=held,
        train_species=sorted(set(tr_df["species_name"]) - held_set),
        train_records=len(tr_df), test_records=len(te_df),
        train_isolates=int(tr_df["sample_id"].nunique()), test_isolates=int(te_df["sample_id"].nunique()),
        train_R=float(tr_df["label"].mean()), test_R=float(te_df["label"].mean()),
        test_frac_records=len(te_df) / total)
    json.dump(summary, open(out / "species_split_summary.json", "w"), indent=1)
    print("[species] SUMMARY:", json.dumps(summary, indent=1), flush=True)
    print(f"[species] wrote {out}/split_antibiogram_species_{{train,test}}.parquet", flush=True)


if __name__ == "__main__":
    main()
