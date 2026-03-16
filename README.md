# 🎾 Tennis Match Prediction with Temporal Graph Networks

> **ATP match outcome prediction using dynamic player graphs — 95k matches, 1991–2024**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?logo=pytorch)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-TGN-purple)](https://pyg.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU-20BEFF?logo=kaggle)](https://kaggle.com)

---

## Overview

This project applies **Temporal Graph Networks (TGN)** to predict ATP tennis match outcomes.  
Players are modeled as **nodes**, matches as **temporal edges** — giving the model access to the full relational and temporal history of each player.

The key design choice: rigorous **walk-forward evaluation** (train on past years, test on future year) to avoid any temporal data leakage — the only honest evaluation protocol for time-series sports data.

---

## Results — Walk-Forward Evaluation (2020–2024)

Each model is trained on all data prior to the test year, then evaluated on that year only.

| Year | ELO AP | XGBoost AP | **TGN AP** | TGN Precision |
|------|--------|------------|------------|---------------|
| 2020 | 0.689  | 0.729      | **0.767**  | 0.685         |
| 2021 | 0.697  | 0.734      | **0.752**  | 0.709         |
| 2022 | 0.729  | 0.761      | **0.783**  | 0.709         |
| 2023 | 0.685  | 0.717      | **0.759**  | 0.689         |
| 2024 | 0.701  | 0.736      | **0.756**  | 0.684         |
| **Avg** | 0.700 | 0.735   | **0.763**  | **0.695**     |

**TGN consistently outperforms XGBoost and ELO every single year** — validating that the temporal graph architecture captures structure beyond hand-crafted features.

---

## Pipeline

### 1. Data preparation
- ATP match data cleaned and processed from 1991 to 2024 (~95k matches)
- **Randomized player sides** (50% swap) to avoid P1/P2 positional bias
- All features computed sequentially — no future information leakage

### 2. Feature engineering
Differential features (P1 − P2) across multiple time windows `K ∈ {3, 5, 10, 25, 50, 100, 200}`:

| Category | Features |
|----------|----------|
| Rating | `ELO_DIFF`, `ELO_SURFACE_DIFF`, `ELO_GRAD_LAST_K_DIFF` |
| Head-to-head | `H2H_DIFF`, `H2H_SURFACE_DIFF` |
| Form | `WIN_LAST_K_DIFF` |
| Serve stats | `P_ACE_LAST_K_DIFF`, `P_DF_LAST_K_DIFF`, `P_1ST_WON_LAST_K_DIFF`, `P_BP_SAVED_LAST_K_DIFF` |
| Context | `ATP_RANK_DIFF`, `ATP_POINTS_DIFF`, `AGE_DIFF`, `HEIGHT_DIFF` |

**67 features total.**

### 3. Graph construction
- Each **player** = a node with a dynamic memory vector
- Each **match** = a directed temporal edge `(p1 → p2, t, msg, y)`

### 4. Model architecture

```
Match (p1, p2, t, features)
        │
        ▼
┌─────────────────────┐
│    TGN Memory        │  ← dynamic state per player s_i ∈ R^128
│    + MessageMLP      │  ← aggregates incoming match messages
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  TimeAwareGNN        │  ← 2 layers × 4 heads TransformerConv
│  (last 25 neighbors) │     edge features = [time_enc ‖ match_msg]
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  SmallWinPredictor   │  ← [z_i ‖ z_j ‖ features] → MLP → P(win)
└─────────────────────┘
```

### 5. Training details

| Hyperparameter | Value |
|----------------|-------|
| Memory dim | 128 |
| Embedding dim | 128 |
| GNN layers × heads | 2 × 4 |
| Optimizer | AdamW (lr=4e-4, wd=1e-4) |
| Scheduler | StepLR (γ=0.9, step=4) |
| Loss | Soft-label BCE (closeness-weighted) |
| Early stopping | patience=10 |
| Warmup events | 5 000 |
| Neighbor size | 25 |

**Soft-label loss**: close matches (7-6 7-6) contribute less signal than dominant wins (6-0 6-0).  
The label is smoothed as `ỹ = 0.5 + (y − 0.5) × (1 − closeness)`.

---

## Baselines

| Model | Avg AP | Avg Accuracy |
|-------|--------|--------------|
| ELO (global) | 0.700 | 64.2% |
| ELO (surface) | 0.688 | 63.3% |
| XGBoost (67 features) | 0.735 | 67.6% |
| **TGN (this work)** | **0.763** | **~70%** |

---

## Repository structure

```
tennis-winner-prediction/
├── notebooks/
│   └── tennis_kaggle_notebook.ipynb   # full pipeline (Kaggle GPU)
├── artifacts/
│   ├── model.pt                       # GNN + predictor weights
│   ├── memory_state.pt                # TGN memory state
│   ├── scaler.pkl                     # feature scaler
│   ├── feature_cols.json              # feature names
│   ├── player_index.json              # player id mapping
│   └── config.json                    # model config
└── README.md
```

---

## Quickstart

```python
# 1. Install dependencies
pip install torch-geometric torch pandas scikit-learn xgboost tqdm

# 2. Run the Kaggle notebook end-to-end
# Add the ATP dataset (atp_matches_YYYY.csv) and run all cells

# 3. Artifacts are saved to /kaggle/working/artifacts/
# Ready for deployment on HF Spaces
```

---

## Key takeaways

- **Walk-forward validation is non-negotiable** for sports prediction — random splits inflate scores significantly
- **TGN adds real value** (+2.8 AP points over XGBoost) by capturing temporal relational dynamics that hand-crafted features cannot fully express
- **ELO surface underperforms global ELO** in disrupted seasons (2021 post-COVID), suggesting surface-specific ratings need minimum match thresholds to be reliable
- **Soft-label loss** on match closeness improves training stability without hurting final metrics

---

## Future work

- Multi-task learning: jointly predict win probability and Elo gain
- Surface-conditional embeddings in the GNN
- Feature enrichment: rest days, tournament draw position, fatigue index
- Deployment on Hugging Face Spaces with live match prediction
