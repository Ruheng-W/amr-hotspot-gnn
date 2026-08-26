# Examples

Small, self-contained examples illustrating the **inputs and outputs** of each
script. The protein example uses a public structure (PDB `1UBQ`, ubiquitin) and
a small illustrative alignment; **no patient-derived data is included**.

## 1. Hotspot detection (`hotspot_detection.py`) — fully runnable

**Inputs** (`examples/hotspot_detection/`):

- `example_protein.pdb` — protein structure (PDB 1UBQ, 76 residues).
- `example_protein.a3m` — MSA in FASTA/A3M format; the first record is the query
  and the rest are homologs (here 1 query + 29 homologs sharing a conserved
  core). Real runs use a UniRef/AlphaFold MSA.

**Command:**

```bash
python hotspot_detection.py \
  --cif examples/hotspot_detection/example_protein.pdb \
  --msa examples/hotspot_detection/example_protein.a3m \
  --save_scores
```

**Outputs** (produced by the command above, saved next to the input):

- `example_protein_hotspots.npy` — 0-based indices of hotspot residues
  (this example: `[9, 61, 62, 8, 45, 5, 46]`).
- `example_protein_site_rates.npy` — per-residue raw conservation
  (Shannon entropy; **lower = more conserved**), one value per residue.
- `example_protein_spatial_rates.npy` — 3D-smoothed conservation per residue.

Load any output with `numpy.load(path)`.

## 2. Training (`train.py`) — manifest schema

Training consumes a **manifest** table plus a cache of per-isolate protein
graphs. `examples/schema/example_manifest.csv` documents the columns:

| column | meaning |
|---|---|
| `sample_id` | unique isolate ID |
| `drug_name` | antibiotic name |
| `drug_smiles` | canonical SMILES of the antibiotic |
| `label` | `0` = susceptible, `1` = resistant |
| `pt_path` | path (relative to `--cache_root`) to the cached isolate graph (`.pt`) |

The real manifests are `.parquet`; this `.csv` only documents the schema. The
cached isolate graphs and clinical labels are not distributed — see the
manuscript's Data Availability statement.

## 3. Interpretability (`interpretability.py`) — panel-map schema

Given a trained checkpoint, a manifest (must contain `pt_path` and `drug_name`),
and a panel map, this script ranks proteins by drug→protein cross-attention and
localizes hotspots. `examples/schema/example_panel_map.csv` documents the panel
map (protein order the model was trained with; the accessions shown are the
case-study proteins from the paper):

| column | meaning |
|---|---|
| `gene_index` | 0-based position of the protein in the model panel |
| `gene_id` | protein identifier (e.g., UniProt accession) |

**Outputs** include a protein-ranking CSV, per-protein hotspot-residue tables, a
`*_node_to_pdb_residue.csv` mapping, a PyMOL `.pml` visualization script, and
drug self-attention heatmaps (PNG).
