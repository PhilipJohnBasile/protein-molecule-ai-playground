# Protein-Ligand Affinity Playground

Lightweight, offline-friendly toolkit for exploring protein–ligand binding affinity prediction and rapid analogue ideation around kinase inhibitors. The repo walks through every stage—from pulling a tiny, free ChEMBL sample to training classical and graph neural models, then generating fragment-based tweaks that satisfy simple property constraints.

---

## Why This Matters

Kinases are high-value drug targets across oncology and immunology. Medicinal chemists routinely evolve known inhibitors to improve potency while keeping drug-like properties in check. Computational triage helps prioritize ideas before synthesis. This playground focuses on:

- **Affinity estimation**: predict pChEMBL values (−log10 activity) from ligand structure and minimal target descriptors.
- **Analogue generation**: propose fragment substitutions (BRICS) that retain key motifs while nudging QED/logP and predicted affinity in a useful direction.
- **Reproducibility**: deterministic seeds, cached datasets, and scripted workflows that run quickly on a laptop.

The project is deliberately small—ideal for experimentation, teaching, or rapid prototyping when full-blown infrastructure is overkill.

---

## Environment Setup

All commands assume **Python 3.11** with RDKit, PyTorch, and PyTorch Geometric available. One reproducible path is micromamba:

```bash
# create environment (CPU-only example)
micromamba create -y -n molplay python=3.11 rdkit=2022.09.5 "pytorch=2.2.*" cpuonly torchvision torchaudio -c conda-forge -c pytorch
micromamba activate molplay

# install project + extras
git clone https://github.com/<your-org>/protein-molecule-ai-playground.git
cd protein-molecule-ai-playground
pip install -e '.[tests]'
pip install torch-geometric==2.5.3
```

> **Note**: Torch/Torch-Geometric wheels must match; adjust versions if you use GPU builds on CUDA-capable hardware. On Apple Silicon (e.g., M1/M2), install the default wheels—PyTorch can use the built-in GPU via the Metal backend (`mps` device), but PyTorch Geometric currently runs CPU-only.

For ad-hoc notebook execution without network-bound kernels, a helper script `scripts/execute_notebook.py` replays cells using the in-process IPython shell.

---

## Repository Layout

```
molplay/                # Python package
  data/                 # fetch & prepare scripts + static cache
  features.py           # featurisers for molecules & targets
  models.py             # RandomForest baseline + GIN model
  train.py              # Typer CLI for training & evaluation
  generate.py           # Typer CLI for BRICS-based analogue design
scripts/                # utility scripts (e.g., notebook executor)
data/raw/               # cached source data (created on demand)
data/clean/             # modelling-ready parquet + metadata
notebooks/              # 01_dataset, 02_train, 03_generate walkthroughs
outputs/                # model artefacts, metrics, generated analogues
tests/                  # pytest-based sanity checks
```

---

## Data Pipeline

1. **Fetch** (offline sample provided):
   ```bash
   python -m molplay.data.fetch --target-family kinase --max-records 5000
   ```
   - Loads `molplay/data/static/chembl_kinase_sample.csv` (20 curated kinase ligands) when the network is unavailable.
   - Writes `data/raw/chembl_kinase_activities.parquet` plus a `.json` manifest.

2. **Prepare & enrich**:
   ```bash
   python -m molplay.data.prepare
   ```
   - Cleans NaNs and enforces `pChEMBL ≥ 5`.
   - Calculates QED, logP (Crippen), and synthetic accessibility proxy (`sa_score`).
   - Keeps minimal target meta (`target_seq_len`) sourced from the static table.
   - Outputs `data/clean/train.parquet` with schema:

     | column | description |
     |--------|-------------|
     | `chembl_id` | ligand identifier |
     | `smiles` | canonical SMILES |
     | `target_id` | ChEMBL target ID |
     | `target_seq_len` | target sequence length (approximate) |
     | `pChEMBL` | −log10 activity |
     | `qed`, `logp`, `sa_score` | RDKit-derived heuristics |

   - Metadata saved to `data/clean/train_meta.json` (record counts, target list, value ranges).

Deterministic seeds ensure identical splits and shuffles across runs.

---

## Feature Engineering (`molplay/features.py`)

- **Molecular graphs**: RDKit → PyTorch Geometric `Data` objects with rich atom/bond features (atom type, degree, valence, chirality, hybridisation; bond type, conjugation, ring membership).
- **Fingerprints**: Morgan (radius 2, 1024 bits) for classical baselines.
- **Target encoding**: default one-hot per target, optionally switchable to sequence-length buckets.

`molplay.utils` provides extra utilities—logging, synthetic accessibility scoring, deterministic scaffolding, and Bemis–Murcko scaffold splitting.

---

## Model Training (`molplay.train` CLI)

```bash
python -m molplay.train --model both --epochs 40 --batch-size 8 --seed 42
```

- **Split strategy**: 70/15/15 Bemis–Murcko scaffold split via `molplay.utils.scaffold_split`. This guards against overfitting to close analogues and better reflects prospective medicinal chemistry campaigns.
- **Baseline**: Scikit-learn `RandomForestRegressor` on concatenated features (Morgan FP + target one-hot + scaled target length + physchem descriptors).
- **GNN**: 3-layer GIN with batch norm and Dropout, global sum pooling, and a target-feature projection pathway.
- **Metrics**: MAE & R² computed separately on train/valid/test sets and persisted in `outputs/metrics.json`.
- **Artifacts**: RandomForest bundle (`baseline_latest.joblib`) stores the fitted estimator plus metadata needed for inference; GNN state dict saved to `gnn_latest.pt` with feature config.

### Reference Metrics (scaffold test set)

Values from a representative CPU run on the packaged 18-point dataset:

| Model | Inputs | Test MAE ↓ | Test R² ↑ |
|-------|--------|------------|-----------|
| RandomForest | Morgan(1024) ⊕ target one-hot ⊕ seq len ⊕ {QED, logP, SA} | 0.29 | -2.16 |
| GIN (3-layer) | Atom/bond graph ⊕ target features | 0.55 | -19.31 |

> **Interpretation**: With only 18 molecules, R² is noisy/negative—expected for such tiny datasets. The pipeline is intended as a template; plug in a larger slice of ChEMBL for meaningful accuracy.

During training the CLI prints epoch progress for the GNN and summarises final metrics.

---

## Analogue Generation (`molplay.generate` CLI)

```bash
python -m molplay.generate \
  --smiles "COc1ccc2nc(Nc3ccc(NC(=O)CO)cc3)ncc2c1OC" \
  --top-k 5
```

Workflow:
1. Load the latest baseline bundle (`outputs/models/baseline_latest.joblib`).
2. Default target inferred from the bundle if none is provided.
3. Apply RDKit BRICS decomposition to the seed, rebuild fragments (with shuffled combinations) while deduplicating canonical SMILES.
4. Filter by QED ≥ 0.4 and logP ≤ 5.0.
5. Score survivors using the baseline regressor; retain top-k.
6. Persist results to `outputs/generated.csv` with columns:
   - `seed_smiles`, `candidate_smiles`, `target_id`
   - `qed`, `logp`, `sa_score`
   - `predicted_pChEMBL`

Example output:
```
seed_smiles,candidate_smiles,target_id,qed,logp,sa_score,predicted_pChEMBL
COc1ccc2nc(...),COc1ccc(-c2ccc(OC)cc2)cc1,CHEMBL1824,0.7795,3.3708,2.1931,7.5175
...
```

---

## Notebooks

1. **`01_dataset.ipynb`** – exploratory data analysis; executed copy saved as `01_dataset.executed.ipynb` for quick review.
2. **`02_train.ipynb`** – hands-on training session for both models, visualising learning curves and errors. (Run locally with a full Jupyter kernel.)
3. **`03_generate.ipynb`** – interactive analogue generation playground, including property histograms and top-ranked structures.

Use the provided micromamba environment (or your own setup) and launch Jupyter as usual:
```bash
micromamba activate molplay
jupyter lab
```

If the sandbox blocks socket creation, replay notebooks via:
```bash
python scripts/execute_notebook.py notebooks/02_train.ipynb notebooks/02_train.executed.ipynb
```

---

## Acceptance Checklist

| Goal | Command | Expected Output |
|------|---------|-----------------|
| Prepare clean dataset | `python -m molplay.data.fetch --target-family kinase`<br>`python -m molplay.data.prepare` | `data/clean/train.parquet` with columns `chembl_id, smiles, target_id, target_seq_len, pChEMBL, qed, logp, sa_score` |
| Train models | `python -m molplay.train --model both` | Console prints MAE/R² for baseline + GNN, artefacts in `outputs/models/`, metrics in `outputs/metrics.json` |
| Generate analogues | `python -m molplay.generate --smiles <seed>` | `outputs/generated.csv` containing candidate SMILES, properties, predicted affinity |
| Validate utilities | `pytest` | All tests pass (featuriser, fingerprint, scaffold split) |
| Explore notebooks | Open notebooks/ in Jupyter or execute via helper script | Visual walkthroughs matching CLI flows |

---

## Testing & Quality Assurance

- Core featuriser/scaffold logic validated by `tests/test_features.py`:
  ```bash
  pytest
  ```
- Deterministic seeds (`molplay.utils.set_global_seed`) ensure repeatable splits, model initialisation, and BRICS sampling.
- Metrics and generated artefacts are written to disk for auditability.

---

## Extending the Playground

- **Bigger datasets**: Adjust `--max-records` and point the fetch script at actual ChEMBL web services or dumps; pipeline scales to a few thousand rows comfortably.
- **Richer target descriptors**: Incorporate sequence embeddings, domain annotations, or structural properties from PDB for more nuanced models.
- **Alternative models**: Swap in XGBoost, LightGBM, or more advanced GNN architectures (e.g., GAT, D-MPNN) by extending `molplay/models.py` and the CLI.
- **Advanced generation**: Replace BRICS with matched molecular pair analysis, reaction-based enumerations, or reinforcement learning loops.

---

## Safety & Ethical Considerations

- Predictions are approximations; **never** treat them as substitutes for experimental validation.
- Generated analogues are vetting by simple heuristics (QED/logP/SA). They may be unstable, unsynthesizable, or unsafe.
- Respect intellectual property and regulatory guidelines when using ChEMBL-derived data.
- Avoid dual-use applications; the toolkit is meant for educational and benign research purposes only.

---

## Troubleshooting

- `ModuleNotFoundError: rdkit`: Ensure you installed the RDKit build matching Python 3.11.
- `torch-geometric` operator errors: reinstall with wheels tailored to your PyTorch build (CPU vs GPU).
- GNN training divergence: increase data size, tune `--epochs`, `--lr`, or shrink hidden dimensions for tiny datasets.
- Notebook execution blocked (permission errors): run locally or via the provided IPython executor.

---

## License & Data Usage

The code is distributed for educational use. Activity data stems from publicly available ChEMBL records; consult the [ChEMBL licence](https://www.ebi.ac.uk/chembl/about#license) for downstream obligations.

Enjoy exploring protein–ligand modelling with a nimble, extensible toolkit!
