# AMR Hotspot-GNN

This repository provides our pipeline for **interpretable antimicrobial resistance (AMR) prediction** by integrating:

- **Drug representation** from SMILES (Transformer encoder)
- **Protein representation** from 3D structure graphs (GVP-GNN)
- **Structure-aware conserved hotspots** computed from **MSA + protein structure** (CONSTRUCT-like)

Our key idea is to first locate **evolutionarily conserved and spatially clustered residues** (hotspots) on each protein structure, then use a **drug→protein cross-attention** module to produce both predictions and **human-interpretable importance scores** (which proteins matter, and where the important residues are).

---

## Hotspot Detection (MSA + Structure)

**Script:** `hotspot_detection.py`

This script detects conserved structural hotspots for a single protein:
1. Load an MSA (`.a3m`, `.fasta`, or AF3 `data.json`)
2. Compute residue-wise conservation using **Shannon entropy**, then **Z-score** it (**lower = more conserved**)
3. Smooth conservation in 3D by distance-weighted neighbors
4. Scan radius **1–20 Å**, pick the best radius by statistical separation, and output hotspot residue indices

### Run
```bash
python hotspot_detection.py \
  --cif path/to/protein.pdb \
  --msa path/to/protein.a3m \
  --save_scores
```
### Outputs
*_hotspots.npy: hotspot residue indices (0-based)
*_spatial_rates.npy (optional): smoothed spatial conservation scores
*_site_rates.npy (optional): raw per-residue conservation scores

## Training
**Script:** `train.py`
This script trains the AMR predictor:
Drug encoder: SMILES Transformer
Protein encoder: GVP-GNN over protein structure graphs
Fusion: multi-head cross-attention from drug embedding to protein embeddings (plus gene presence mask)
Data loading: SmartMemoryDataset caches unique isolate graphs in RAM to reduce repeated I/O
### Run
```bash
python train.py \
  --cache_root /path/to/cache_root \
  --train_manifest /path/to/train.parquet \
  --val_manifest /path/to/val.parquet \
  --n_genes 333 \
  --out_dir /path/to/output \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --load_to_ram
```

## Interpretability
**Script:** `interpretability.py`
This script explains predictions for a given drug_name:
Saves a drug self-attention heatmap and SMILES token importance
Ranks proteins by drug→protein cross-attention
Maps *_hotspots.npy indices back to PDB residues
Generates a PyMOL script to visualize hotspots on top-ranked proteins
### Run
```bash
python interpretability.py \
  --ckpt /path/to/checkpoint.pt \
  --n_genes 333 \
  --manifest /path/to/manifest.parquet \
  --cache_root /path/to/cache_root \
  --panel_map_csv /path/to/panel_map.csv \
  --drug_name YOUR_DRUG_NAME \
  --structure_dir /path/to/structures \
  --hotspot_dir /path/to/hotspots \
  --map_dir /path/to/node_maps \
  --out_dir /path/to/out \
  --top_proteins 10
```

## What Each File Does
hotspot_detection.py
ConservationCalculator: loads MSA and computes entropy-based conservation scores
ConstructAlgo: parses structure (PDB/mmCIF), performs 3D smoothing + radius scan, outputs hotspot indices

train.py
SmilesTokenizer: character-level SMILES tokenizer and vocabulary builder
DrugTransformer: Transformer encoder for drug SMILES
ProteinGVPEncoder: GVP-based encoder for protein structure graphs
AMRPredictor: drug–protein cross-attention fusion + binary classifier head
SmartMemoryDataset: shared in-RAM graph bank to avoid re-loading duplicate isolate graphs

interpretability.py
Captures drug self-attention (layer/head selectable) and exports token importance
Computes drug→protein cross-attention for protein ranking
Builds node→PDB residue mapping CSV and exports hotspot residue tables
Writes a PyMOL script (.pml) for hotspot visualization
