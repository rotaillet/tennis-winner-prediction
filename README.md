# 🎾 Tennis Match Prediction with Temporal Graph Networks (TGN)

## 📌 Project Overview
This project explores the use of **Temporal Graph Networks (TGN)** to predict the winner of ATP tennis matches.  
It brings together two passions: **tennis and artificial intelligence** 🧠.

Unlike traditional approaches such as Elo ratings or tabular models (Logistic Regression, XGBoost), the idea is to exploit both the **temporal** and **relational** structure of the data by modeling players and matches as a dynamic graph.

---

## ⚙️ Pipeline
1. **Data preparation**
   - Cleaning and feature engineering from ATP match data  
   - Differential features such as:  
     `ELO_DIFF`, `ATP_POINTS_DIFF`, `H2H_DIFF`, `P_ACE_LAST_50_DIFF`, etc.  

2. **Graph construction**
   - Each player = a node  
   - Each match = a directed edge `(p1_id → p2_id)` with:  
     - timestamp `t`  
     - message `msg` (match features)  
     - label `y` (match result)  

3. **Model architecture**
   - **TGNMemory**: dynamic player memory  
   - **MultiLayerTimeAwareGNN**: time-aware GNN using TransformerConv  
   - **SmallWinPredictor**
 classification head predicting match outcome
:<img width="1763" height="61" alt="tgn_model_diagram" src="https://github.com/user-attachments/assets/6eb8f9a0-6906-4570-8e52-b6d3dfffa003" />


4. **Training setup**
   - Optimizer: `AdamW`  
   - Loss: `BCEWithLogitsLoss`  
   - Regularization: `Dropout`, `Weight Decay`, `DropEdge` (optional)  
   - Early stopping

---

## 📊 Results
- Achieved accuracy: **~73–74%**, close to surface-specific Elo benchmarks.  
- While not a breakthrough, the project provided key **insights**:  
  - Label leakage must be carefully avoided  
  - Temporal GNNs are powerful but prone to **overfitting** and costly to train  
  - Evaluation protocol (temporal split vs random split) makes a huge difference  

---

## 🚀 Future Work
- **Multi-task learning**: jointly predict win probability and score closeness / Elo gain  
- **More robust validation**: larger temporal splits, panel by year or surface  
- **Feature enrichment**: rest days, fatigue, weather, tournament conditions  
- **Stronger baselines**: Logistic Regression, MLP, XGBoost

---

