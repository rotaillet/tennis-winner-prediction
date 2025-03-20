import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # On s'appuie ici sur torch_geometric pour la convolution sur graphe
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import compute_player_differences,build_mappings,build_player_history,split_last_match

##############################################
# 1. MODULE DE MÉMOIRE POUR LES NŒUDS
##############################################

class MemoryModule(nn.Module):
    """
    Module qui stocke la mémoire (état) pour chacun des nœuds.
    """
    def __init__(self, num_nodes: int, memory_dim: int):
        super(MemoryModule, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        # On initialise la mémoire à zéro pour tous les nœuds.
        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
    
    def reset_memory(self, device):
        self.memory = torch.zeros(self.num_nodes, self.memory_dim, device=device)
    
    def get_memory(self, node_indices: torch.LongTensor):
        return self.memory[node_indices]
    
    def update_memory(self, node_indices: torch.LongTensor, new_memory: torch.FloatTensor):
        self.memory[node_indices] = new_memory

##############################################
# 2. CELLULE TGN (avec message passing + GRU)
##############################################

class TGNCell(nn.Module):
    """
    Cellule TGN minimaliste.
    
    Elle combine :
      - Une opération de message passing (ici via un GCN) sur la concaténation des features d'entrée et de la mémoire,
      - Une mise à jour de la mémoire avec une cellule GRU.
    
    Args:
        in_channels (int): Dimension des features d'entrée.
        memory_dim (int): Dimension de la mémoire (état de chaque nœud).
        out_channels (int): Dimension de la sortie du GCN (par exemple, une représentation intermédiaire).
    """
    def __init__(self, in_channels: int, memory_dim: int, out_channels: int):
        super(TGNCell, self).__init__()
        # On combine les features d'entrée et la mémoire : dimension totale = in_channels + memory_dim.
        self.gcn = GCNConv(in_channels + memory_dim, out_channels)
        # La cellule GRU met à jour la mémoire à partir du message issu du GCN.
        self.gru = nn.GRUCell(out_channels, memory_dim)
    
    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor, memory: torch.FloatTensor):
        """
        Args:
            x: Features d'entrée pour tous les nœuds, shape [num_nodes, in_channels].
            edge_index: Indices des arêtes (interactions) du graphe.
            memory: Mémoire actuelle des nœuds, shape [num_nodes, memory_dim].
        
        Retourne:
            new_memory: Mémoire mise à jour, shape [num_nodes, memory_dim].
            message: La sortie du GCN (peut servir d'embedding temporaire), shape [num_nodes, out_channels].
        """
        # Concaténation des features et de la mémoire
        combined = torch.cat([x, memory], dim=1)
        message = self.gcn(combined, edge_index)
        # Mise à jour de la mémoire via GRU
        new_memory = self.gru(message, memory)
        return new_memory, message

##############################################
# 3. TEMPORAL GRAPH NETWORK (TGN)
##############################################

class TGN(nn.Module):
    """
    Implémentation minimale d'un Temporal Graph Network.
    
    Ce modèle met à jour les représentations des nœuds au fil des interactions (événements).
    """
    def __init__(self, num_nodes: int, in_channels: int, memory_dim: int, out_channels: int):
        """
        Args:
            num_nodes (int): Nombre total de nœuds.
            in_channels (int): Dimension des features d'entrée pour chaque nœud.
            memory_dim (int): Dimension de la mémoire.
            out_channels (int): Dimension de sortie de la cellule TGN.
        """
        super(TGN, self).__init__()
        self.memory_module = MemoryModule(num_nodes, memory_dim)
        self.cell = TGNCell(in_channels, memory_dim, out_channels)
    
    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor):
        """
        Args:
            x: Features d'entrée pour tous les nœuds, shape [num_nodes, in_channels].
            edge_index: Indices des arêtes pour l'événement courant.
        
        Retourne:
            message: La sortie de la cellule (représentation temporaire) pour tous les nœuds.
            new_memory: La mémoire mise à jour pour tous les nœuds.
        """
        memory = self.memory_module.memory.detach()
        new_memory, message = self.cell(x, edge_index, memory)
        return message, new_memory

##############################################
# 4. MODÈLE DE PRÉDICTION DE MATCH AVEC TGN
##############################################

class TGNMatchPredictor(nn.Module):
    """
    Ce modèle utilise le TGN pour mettre à jour les représentations des joueurs et
    combine ensuite ces représentations (pour deux joueurs, plus un embedding de tournoi et
    des features statiques du match) afin de prédire le résultat d'un match de tennis.
    """
    def __init__(self, 
                 num_nodes: int,
                 in_channels: int,
                 memory_dim: int,
                 tgcn_out_channels: int,
                 num_tournois: int,
                 tournament_embedding_dim: int,
                 static_feature_dim: int,
                 classifier_hidden_dim: int = 256,
                 dropout: float = 0.3):
        super(TGNMatchPredictor, self).__init__()
        # Le TGN qui met à jour les représentations des nœuds.
        self.tgn = TGN(num_nodes, in_channels, memory_dim, tgcn_out_channels)
        # Embedding pour le tournoi.
        self.tournoi_embedding = nn.Embedding(num_tournois, tournament_embedding_dim)
        # Le classifieur combine l'embedding du joueur 1, du joueur 2, l'embedding du tournoi
        # et les features statiques du match.
        classifier_input_dim = 2 * tgcn_out_channels + tournament_embedding_dim + static_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 4, 2)  # 2 classes : victoire de j1 ou victoire de j2.
        )
    
    def forward(self,
                initial_node_features: torch.FloatTensor,  # [num_nodes, in_channels]
                edge_index: torch.LongTensor,              # Indices de l'interaction (match courant)
                player1_idx: torch.LongTensor,             # Indice(s) du joueur 1 (batch)
                player2_idx: torch.LongTensor,             # Indice(s) du joueur 2 (batch)
                tournoi_idx: torch.LongTensor,             # Indice(s) du tournoi (batch)
                static_feat: torch.FloatTensor             # Features statiques du match [batch, static_feature_dim]
               ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Retourne:
            logits: Logits de prédiction, shape [batch, 2].
            H_new: État (représentations) mis à jour pour tous les nœuds.
        """
        # On met à jour la représentation de tous les nœuds via le TGN.
        message, H_new = self.tgn(initial_node_features, edge_index)
        # Extraction des embeddings pour les deux joueurs impliqués dans le match.
        embed_p1 = H_new[player1_idx]  # [batch, tgcn_out_channels]
        embed_p2 = H_new[player2_idx]  # [batch, tgcn_out_channels]
        # Récupération de l'embedding du tournoi.
        embed_tournoi = self.tournoi_embedding(tournoi_idx)  # [batch, tournament_embedding_dim]
        # Concaténation des embeddings et des features statiques.
        combined = torch.cat([embed_p1, embed_p2, embed_tournoi, static_feat], dim=1)
        logits = self.classifier(combined)
        return logits, H_new

##############################################
# 5. Exemple de Pipeline d'Entraînement
##############################################

# Pour cet exemple, nous allons créer un dataset fictif.
class TennisMatchDataset(Dataset):
    def __init__(self, df, player_to_idx, tournoi_to_idx):
        # On trie par date pour pouvoir utiliser l'historique
        self.df = df.sort_values("date").reset_index(drop=True)
        self.player_to_idx = player_to_idx
        self.tournoi_to_idx = tournoi_to_idx

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        current_date = row["date"]

        # Calcul de la différence de performances entre joueurs (cette fonction doit être définie ailleurs)
        diff = compute_player_differences(row)  # À définir selon vos critères

        # Fonction interne pour calculer les features head-to-head
        def compute_head_to_head(p1, p2):
            df_h2h = self.df[((self.df["j1"] == p1) & (self.df["j2"] == p2)) | 
                             ((self.df["j1"] == p2) & (self.df["j2"] == p1))]
            df_h2h = df_h2h[df_h2h["date"] < current_date]
            total = len(df_h2h)
            if total == 0:
                win_ratio = 0.5
            else:
                wins = (df_h2h["winner"] == p1).sum()
                win_ratio = wins / total
            return np.array([total, win_ratio], dtype=np.float32)
        
        # Calcul des features head-to-head et combinaison avec les différences
        head2head_features = compute_head_to_head(row["j1"], row["j2"])
        combined_static = np.concatenate([head2head_features, diff])
        
        # Conversion des identifiants en indices numériques
        player1_idx = self.player_to_idx[row["j1"]]
        player2_idx = self.player_to_idx[row["j2"]]
        tournoi_idx = self.tournoi_to_idx[row["tournament"]]
        
        # Construction de l'edge_index pour le match : tensor de forme [2, 1]
        edge_index = torch.tensor([[player1_idx], [player2_idx]], dtype=torch.long)
        
        # Définition de la cible : 0 si j1 gagne, 1 si j2 gagne
        if row["winner"] == row["j1"]:
            target = 0
        elif row["winner"] == row["j2"]:
            target = 1
        else:
            raise ValueError(f"Nom de gagnant inattendu : {row['winner']}")
        
        return {
            "edge_index": edge_index,
            "static_feat": torch.tensor(combined_static, dtype=torch.float),
            "target": torch.tensor(target, dtype=torch.long),
            "player1_idx": torch.tensor(player1_idx, dtype=torch.long),
            "player2_idx": torch.tensor(player2_idx, dtype=torch.long),
            "tournoi_idx": torch.tensor(tournoi_idx, dtype=torch.long),
            "date": row["date"]
        }




df = pd.read_csv("data/all_features.csv")
player_to_idx, tournoi_to_idx = build_mappings(df)
num_players = len(player_to_idx)
num_tournois = len(tournoi_to_idx)
history = build_player_history(df)
train_df, test_df = split_last_match(df)

# Instanciation du dataset et du DataLoader
train_dataset = TennisMatchDataset(train_df,player_to_idx,tournoi_to_idx)
test_dataset = TennisMatchDataset(test_df,player_to_idx,tournoi_to_idx)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False,num_workers=10)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False,num_workers=10)

# Paramètres du modèle
in_channels = 4               # Dimension des features initiales des joueurs (par exemple, rank et points)
tgcn_out_channels = 64         # Dimension de sortie de la cellule TGN
memory_dim = 64                # Dimension de la mémoire dans le TGN
# Ici, dans notre TGN "maison", on utilise la même dimension pour la mémoire et la sortie du GCN.
num_tournois_total = num_tournois
tournament_embedding_dim = 16   # Dimension de l'embedding pour le tournoi
static_feature_dim = 6          # Dimension des features statiques du match


# Instanciation du modèle
model = TGNMatchPredictor(num_nodes=num_players,
                           in_channels=in_channels,
                           memory_dim=memory_dim,
                           tgcn_out_channels=tgcn_out_channels,
                           num_tournois=num_tournois_total,
                           tournament_embedding_dim=tournament_embedding_dim,
                           static_feature_dim=static_feature_dim,
                           classifier_hidden_dim=256,
                           dropout=0.3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Initialisation d'un dictionnaire pour stocker l'historique
player_dynamic_features = {player: [] for player in player_to_idx.keys()}

# On parcourt toutes les lignes du DataFrame
for _, row in df.iterrows():
    # Supposons que les colonnes "date", "j1", "j1_rank", "j1_points" existent,
    # ainsi que "j2", "j2_rank", "j2_points"
    player_dynamic_features[row["j1"]].append((row["date"], row["rank1"], row["point1"],row["elo_j1"],row["age1"]))
    player_dynamic_features[row["j2"]].append((row["date"], row["rank2"], row["point2"],row["elo_j2"],row["age2"]))

# On trie les listes pour chaque joueur par date croissante
for player in player_dynamic_features:
    player_dynamic_features[player] = sorted(player_dynamic_features[player], key=lambda x: x[0])
def get_player_features_at_date(player, current_date, player_dynamic_features, default_features=np.array([3000, 0,1500,25], dtype=np.float32)):
    """
    Retourne les caractéristiques [rank, points] du joueur avant la date current_date.
    Si aucune donnée n'est disponible, renvoie default_features.
    """
    features = default_features
    for date, rank, points,elos,ages in player_dynamic_features[player]:
        # On suppose ici que les dates sont comparables (par exemple, sous forme de string ISO ou de datetime)
        if date < current_date:
            features = np.array([rank, points, elos, ages], dtype=np.float32)
        else:
            break
    return features

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
from torch_geometric.utils import add_self_loops

# Dans la boucle d'entraînement, après avoir obtenu edge_index :


# Entraînement sur quelques époques
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        # Récupération et passage des données sur le device
        batch_dates = batch['date']  # Par exemple, une liste ou tensor de dates
        current_date = min(batch_dates)  # À adapter selon votre format de date
        
        # Mise à jour dynamique des features pour tous les joueurs
        features_list = []
        for player, idx in sorted(player_to_idx.items(), key=lambda x: x[1]):
            feat = get_player_features_at_date(player, current_date, player_dynamic_features)
            features_list.append(feat)
        dynamic_initial_node_features = torch.tensor(np.stack(features_list), dtype=torch.float).to(device)
    
        edge_index = batch['edge_index'].squeeze(2).transpose(0, 1).to(device) # shape [2, batch] si on empile horizontalement
        edge_index, _ = add_self_loops(edge_index, num_nodes=dynamic_initial_node_features.size(0))
        player1_idx = batch['player1_idx'].to(device)
        player2_idx = batch['player2_idx'].to(device)
        tournoi_idx = batch['tournoi_idx'].to(device)
        static_feat = batch['static_feat'].to(device)
        targets = batch['target'].to(device)
        
        # Passage dans le modèle
        logits, new_memory = model(dynamic_initial_node_features, edge_index,
                          player1_idx, player2_idx, tournoi_idx, static_feat)
        loss = criterion(logits, targets)
        loss.backward()
        with torch.no_grad():
            model.tgn.memory_module.memory.copy_(new_memory.detach())
        optimizer.step()
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss (Entraînement): {avg_loss:.4f}")
    
    # Phase d'évaluation sur le jeu de test
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Test)"):
            batch_dates = batch['date']  # Par exemple, une liste ou tensor de dates
            current_date = min(batch_dates)  # À adapter selon votre format de date
        
        # Mise à jour dynamique des features pour tous les joueurs
            features_list = []
            for player, idx in sorted(player_to_idx.items(), key=lambda x: x[1]):
            
                feat = get_player_features_at_date(player, current_date, player_dynamic_features)
                features_list.append(feat)
            dynamic_initial_node_features = torch.tensor(np.stack(features_list), dtype=torch.float).to(device)
            edge_index = batch['edge_index'].squeeze(2).transpose(0, 1).to(device)
            edge_index, _ = add_self_loops(edge_index, num_nodes=dynamic_initial_node_features.size(0))
            player1_idx = batch['player1_idx'].to(device)
            player2_idx = batch['player2_idx'].to(device)
            tournoi_idx = batch['tournoi_idx'].to(device)
            static_feat = batch['static_feat'].to(device)
            targets = batch['target'].to(device)
            
            logits, new_memory = model(dynamic_initial_node_features, edge_index,
                                       player1_idx, player2_idx, tournoi_idx, static_feat)
            loss = criterion(logits, targets)
            test_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    avg_test_loss = test_loss / len(test_dataloader)
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss (Test): {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}")

print("Entraînement terminé.")

