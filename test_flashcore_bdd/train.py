from tqdm import tqdm
import torch

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer,
                node_features, edge_index, edge_weight, num_epochs=30, device="cpu"):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Train", leave=False)
        for batch in train_progress_bar:
            p1_history = batch["p1_history"].to(device)
            p2_history = batch["p2_history"].to(device)
            static_feat = batch["static_feat"].to(device)
            targets = batch["target"].to(device)
            player1_idx = batch["player1_idx"].to(device)
            player2_idx = batch["player2_idx"].to(device)
            tournoi_idx = batch["tournoi_idx"].to(device)
            
            optimizer.zero_grad()
            outputs = model(p1_history, p2_history, static_feat, player1_idx, player2_idx, tournoi_idx,
                            node_features.to(device), edge_index.to(device), edge_weight.to(device))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * p1_history.size(0)
            train_progress_bar.set_postfix(loss=loss.item())
            
        train_loss = running_loss / len(train_dataloader.dataset)
        test_loss, test_accuracy = test_model(model, test_dataloader, criterion, node_features, edge_index, edge_weight, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")

def test_model(model, dataloader, criterion, node_features, edge_index, edge_weight, device="cpu"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    misclassified_matches = []  # Liste pour stocker les erreurs

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test", leave=False):
            p1_history = batch["p1_history"].to(device)
            p2_history = batch["p2_history"].to(device)
            static_feat = batch["static_feat"].to(device)
            targets = batch["target"].to(device)
            player1_idx = batch["player1_idx"].to(device)
            player2_idx = batch["player2_idx"].to(device)
            tournoi_idx = batch["tournoi_idx"].to(device)

            outputs = model(p1_history, p2_history, static_feat, player1_idx, player2_idx, tournoi_idx,
                            node_features.to(device), edge_index.to(device), edge_weight.to(device))

            loss = criterion(outputs, targets)
            running_loss += loss.item() * p1_history.size(0)

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Stocker les erreurs
            incorrect_indices = (predicted != targets).nonzero(as_tuple=True)[0]
            for idx in incorrect_indices:
                misclassified_matches.append({
                        "player1_name": batch["player1_name"][idx],
                        "player2_name": batch["player2_name"][idx],
                        "tournoi_name": batch["tournoi_name"][idx],
                        "true_label": targets[idx].item(),
                        "predicted_label": predicted[idx].item()
                })

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.4f}")
    print(f"Nombre de matchs mal classés : {len(misclassified_matches)}")

    # Affichage des erreurs
    print("\n--- Matchs mal classés ---")
    for error in misclassified_matches[0:3]:  # Afficher les 10 premières erreurs
        print(f"Player1: {error['player1_name']} | Player2: {error['player2_name']} | Tournoi: {error['tournoi_name']}")
        print(f"   ➝ Vraie classe: {error['true_label']} | Prédiction: {error['predicted_label']}\n")

    return avg_loss, accuracy, misclassified_matches  # Retourne aussi la liste des erreurs


import torch.nn.functional as F
def test_model2(model, dataloader, criterion, graph_data, device="cpu"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_softmax_outputs = []
    all_predictions = []  # Pour stocker les prédictions
    all_targets = []      # Pour stocker les vérités

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test", leave=False):
            static_feat = batch["static_feat"].to(device)
            targets = batch["target"].to(device)
            player1_idx = batch["player1_idx"].to(device)
            player2_idx = batch["player2_idx"].to(device)
            tournoi_idx = batch["tournoi_idx"].to(device)
            player1_seq = batch["player1_seq"].to(device)
            player2_seq = batch["player2_seq"].to(device)
            
            outputs = model(static_feat, player1_idx, player2_idx, tournoi_idx, graph_data, player1_seq, player2_seq)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            softmax_outputs = F.softmax(outputs, dim=1)
            all_softmax_outputs.append(softmax_outputs)

            # Obtenir la prédiction (classe avec la plus haute probabilité)
            _, predicted = torch.max(outputs, 1)
            all_predictions.append(predicted)
            all_targets.append(targets)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    # Concaténer tous les batchs
    all_softmax_outputs = torch.cat(all_softmax_outputs, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculer la probabilité maximale pour chaque prédiction
    max_probs, _ = torch.max(all_softmax_outputs, dim=1)
    # Créer un masque pour les prédictions avec une confiance > 80%
    high_conf_mask = max_probs > 0.61

    # Sélectionner les prédictions et cibles correspondantes
    high_conf_predictions = all_predictions[high_conf_mask]
    high_conf_targets = all_targets[high_conf_mask]

    # Compter le nombre de cas où la prédiction était correcte
    high_conf_correct = (high_conf_predictions == high_conf_targets).sum().item()
    number_high_conf = high_conf_mask.sum().item()

    print(f"Nombre de prédictions avec confiance >80%: {number_high_conf}")
    print(f"Nombre de victoires réelles parmi ces prédictions: {high_conf_correct}")
    print(f"statistique associé à ces predictions : {high_conf_correct/number_high_conf *100:.2f}%")
    return avg_loss, accuracy, all_softmax_outputs

