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

