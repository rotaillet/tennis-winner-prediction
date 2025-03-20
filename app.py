from flask import Flask, request, jsonify, render_template
import pandas as pd
import datetime
import torch
from utils import (
    normalize_columns, build_mappings, build_node_features, compute_player_differences,get_days_since_last_match,
    build_player_graph_with_weights, build_player_history, split_last_match,build_player_graph_with_weights_recent
)
from dataset import TennisMatchDataset2
import numpy as np
from model import TennisMatchPredictor

from torch.utils.data import Dataset, DataLoader
app = Flask(__name__)

# Charger le modèle pré-entraîné

STATIC_FEATURE_DIM = 34   
HIST_FEATURE_DIM = 9
d_model = 256
gnn_hidden = 256
num_gnn_layers = 12
dropout = 0.3
df = pd.read_csv("data/all_features2.csv", parse_dates=["date"])


player_to_idx, tournoi_to_idx = build_mappings(df)
history = build_player_history(df)
train_df, test_df = split_last_match(df)
# Supposons que df contient toutes les rencontres, player_to_idx et tournoi_to_idx sont vos mappings.
edge_index, edge_weight = build_player_graph_with_weights_recent(df, player_to_idx, lambda_=0.001)
node_features = build_node_features(df, player_to_idx)
num_players = len(player_to_idx)
num_tournois = len(tournoi_to_idx)

last_match_list = []

# Pour itérer sur chaque match du train_df
for idx, row in train_df.iterrows():
    current_date = row["date"]
    

    # 4. Temps depuis le dernier match pour chaque joueur
    last_match_p1 = get_days_since_last_match(history, row["j1"], current_date)
    last_match_p2 = get_days_since_last_match(history, row["j2"], current_date)
    last_match_list.append(last_match_p1)
    last_match_list.append(last_match_p2)

# Calcul des moyennes et écarts-types avec une petite constante pour éviter la division par zéro
epsilon = 1e-6
norm_params = {

    
    "last_match_mean": np.mean(last_match_list),
    "last_match_std": np.std(last_match_list) + epsilon,
}
from torch_geometric.data import Data
graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)

model = TennisMatchPredictor(STATIC_FEATURE_DIM, num_players, num_tournois, d_model, gnn_hidden,HIST_FEATURE_DIM, num_gnn_layers, dropout,seq_length=5)

state_dict = torch.load('model/best_model84.pth', map_location=torch.device('cpu'))
device = "cuda" if torch.cuda.is_available() else "cpu"
graph_data = graph_data.to(device)
model.load_state_dict(state_dict)
model.eval()

# Charger le CSV dans un DataFrame
# On peut charger le fichier une seule fois au démarrage si le CSV n'est pas mis à jour en temps réel.
matches_df = pd.read_csv('data/all_features2.csv', parse_dates=['date'])



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/matches', methods=['GET'])
def get_matches():
    # Récupérer la date passée en paramètre GET (format 'YYYY-MM-DD')
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'Date non spécifiée'}), 400

    try:
        date_limite = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Format de date invalide, utilisez YYYY-MM-DD'}), 400

    # Filtrer les matchs dont la date est supérieure ou égale à la date_limite
    filtered_matches = matches_df[matches_df['date'] >= date_limite]

    # Convertir les résultats en liste de dictionnaires
    # On conserve les colonnes pertinentes : id, date, joueur1 et joueur2
    matches_list = filtered_matches[['href', 'date', 'j1', 'j2']].to_dict(orient='records')
    
    # Convertir la date en chaîne de caractères
    for match in matches_list:
        match['date'] = match['date'].strftime('%Y-%m-%d')

    return jsonify({'matches': matches_list})


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json(force=True)

    match_id = data.get('match_id') 

    if match_id is None:
        return jsonify({'error': 'Aucun identifiant de match fourni.'}), 400

    # Rechercher le match correspondant dans le DataFrame
    match_row = matches_df[matches_df['href'] == match_id]
    if match_row.empty:
        return jsonify({'error': 'Match introuvable.'}), 404
    match_row.reset_index(inplace=True,drop=True)
    player1_name = match_row['j1'][0]
    player2_name = match_row['j2'][0]

    test_dataset = TennisMatchDataset2(match_row,history,player_to_idx,tournoi_to_idx,norm_params)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=1)
    # Prétraiter les données du match
    

    for batch in test_dataloader:
        # Déballer les éléments du batch
        static_feat = batch["static_feat"]
        player1_seq = batch["player1_seq"]
        player2_seq = batch["player2_seq"]
        player1_idx = batch["player1_idx"]
        player2_idx = batch["player2_idx"]
        tournoi_idx = batch["tournoi_idx"]
        
        
        # Effectuer la prédiction avec le modèle
        with torch.no_grad():
            # Veillez à transférer les tenseurs sur le même device que votre modèle
            static_feat = static_feat.to(device)
            player1_seq = player1_seq.to(device)
            player2_seq = player2_seq.to(device)
            player1_idx = player1_idx.to(device)
            player2_idx = player2_idx.to(device)
            tournoi_idx = tournoi_idx.to(device)
            # graph_data_batch est déjà un Data, assurez-vous qu'il est sur le device
            global_graph = graph_data.to(device)
            
            output = model(static_feat, player1_idx, player2_idx, tournoi_idx, global_graph, player1_seq, player2_seq)
            output = F.softmax(output, dim=1)
            prob_class0 = output[0][0].item()
            prob_class1 = output[0][1].item()
            

    return jsonify({
        'player1': player1_name,
        'player2': player2_name,
        'prob_class0': prob_class0,
        'prob_class1': prob_class1
    })


if __name__ == '__main__':
    app.run(debug=True)
