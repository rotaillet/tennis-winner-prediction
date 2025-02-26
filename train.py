# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from tennis_model import TennisModelGATTransformer
from data_preprocessing import load_data, build_data

# Chargement et prétraitement des données
df = load_data("test.csv")
data_dict = build_data(df)
edge_index = data_dict["edge_index"]
player_features = data_dict["player_features"]
match_data_tensor = data_dict["match_data_tensor"]
match_features_tensor = data_dict["match_features_tensor"]
labels_tensor = data_dict["labels_tensor"]
match_dynamic_histories = data_dict["match_dynamic_histories"]

print("# Data preprocessing completed")

# Définition du Dataset
class TennisDataset(Dataset):
    def __init__(self, match_data, match_features, labels, dynamic_histories):
        self.match_data = match_data
        self.match_features = match_features
        self.labels = labels
        self.dynamic_histories = dynamic_histories

    def __len__(self):
        return len(self.match_data)

    def __getitem__(self, idx):
        return (self.match_data[idx],
                self.match_features[idx],
                self.labels[idx],
                self.dynamic_histories[idx])

def custom_collate(batch):
    match_data = torch.stack([item[0] for item in batch], dim=0)
    match_features = torch.stack([item[1] for item in batch], dim=0)
    labels = torch.stack([item[2] for item in batch], dim=0)
    dynamic_histories = [item[3] for item in batch]
    p1_hist_list = [dh[0] for dh in dynamic_histories]
    p2_hist_list = [dh[1] for dh in dynamic_histories]
    return match_data, match_features, labels, (p1_hist_list, p2_hist_list)

# Création des datasets train/test
num_matches = len(match_data_tensor)
train_size = int(0.8 * num_matches)
train_indices = list(range(train_size))
test_indices = list(range(train_size, num_matches))

train_dataset = TennisDataset(match_data_tensor[train_indices],
                                match_features_tensor[train_indices],
                                labels_tensor[train_indices],
                                match_dynamic_histories[:train_size])
test_dataset = TennisDataset(match_data_tensor[test_indices],
                               match_features_tensor[test_indices],
                               labels_tensor[test_indices],
                               match_dynamic_histories[train_size:])

batch_size = 256
epochs = 30
tscv = TimeSeriesSplit(n_splits=5)
fold_results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, val_idx) in enumerate(tscv.split(np.arange(num_matches))):
    print(f"\n=== Fold {fold+1} ===")
    train_dataset_fold = TennisDataset(match_data_tensor[train_idx],
                                         match_features_tensor[train_idx],
                                         labels_tensor[train_idx],
                                         [match_dynamic_histories[i] for i in train_idx])
    val_dataset_fold = TennisDataset(match_data_tensor[val_idx],
                                       match_features_tensor[val_idx],
                                       labels_tensor[val_idx],
                                       [match_dynamic_histories[i] for i in val_idx])
    train_loader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    model = TennisModelGATTransformer(
        feature_dim=player_features.size(1),
        hidden_dim=128,
        transformer_dim=128,
        combined_dim=128,
        match_feat_dim=match_features_tensor.size(1),
        dropout_p=0.3
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3,verbose=True)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Fold {fold+1} - Epoch {epoch+1}/{epochs}", leave=False)
        for match_batch, match_feat_batch, label_batch, dyn_hist_batch in progress_bar:
            match_batch = match_batch.to(device)
            match_feat_batch = match_feat_batch.to(device)
            label_batch = label_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(
                player_features.to(device),
                edge_index.to(device),
                dyn_hist_batch,
                match_batch,
                match_feat_batch
            )
            loss = loss_fn(outputs, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * match_batch.size(0)
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader.dataset)
        scheduler.step()
        print(f"Fold {fold+1}, Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for match_batch, match_feat_batch, label_batch, dyn_hist_batch in val_loader:
            match_batch = match_batch.to(device)
            match_feat_batch = match_feat_batch.to(device)
            label_batch = label_batch.to(device)
            outputs = model(
                player_features.to(device),
                edge_index.to(device),
                dyn_hist_batch,
                match_batch,
                match_feat_batch
            )
            predicted = (outputs > 0.5).float()
            correct += (predicted == label_batch).sum().item()
            total += label_batch.size(0)
    val_accuracy = 100 * correct / total
    print(f"Fold {fold+1} - Validation Accuracy: {val_accuracy:.2f}%")
    fold_results.append(val_accuracy)

print("\n=== Cross-validation Results ===")
print(f"Average Accuracy over {tscv.get_n_splits()} folds: {np.mean(fold_results):.2f}%")
