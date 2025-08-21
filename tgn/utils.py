import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,precision_score
from tqdm import tqdm

@torch.no_grad()
def evaluate(loader, memory, gnn, win_pred, full_data, eval_loader_ngh, assoc, device, thresholds, alpha):
    memory.eval()
    gnn.eval()
    win_pred.eval()
    total_loss = 0.0
    total_events = 0
    all_preds = []
    all_trues = []
    all_dates = []

    for batch in tqdm(loader, desc="Evaluating", unit="batch"):
        batch = batch.to(device)
        nodes = torch.cat([batch.src, batch.dst]).unique()
        n_id1, edge_index1, e_id1 = eval_loader_ngh(nodes)
        n_id, edge_index, e_id = eval_loader_ngh(n_id1)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        t_e = full_data.t[e_id].to(device)
        msg_e = full_data.msg[e_id].to(device)
        z = gnn(z, last_update, edge_index, t_e, msg_e)
        match_feats =batch.msg
        logit_win = win_pred(z[assoc[batch.src]], z[assoc[batch.dst]],match_feats)
        y_win = batch.y.view(-1, 1)
        

        loss_win = F.binary_cross_entropy_with_logits(logit_win, y_win)

        loss = loss_win
        total_loss += loss.item() * batch.num_events
        total_events += batch.num_events

        y_pred = logit_win.sigmoid().cpu()
        y_true = batch.y.view(-1,1).cpu()

        all_preds.append(y_pred)
        all_trues.append(y_true)
        all_dates.append(batch.t.cpu())

        msg_with_label = torch.cat([batch.msg], dim=-1)
        memory.update_state(batch.src, batch.dst, batch.t, msg_with_label)
        eval_loader_ngh.insert(batch.src, batch.dst)

    avg_loss = total_loss / total_events
    all_preds = torch.cat(all_preds).numpy()
    all_trues = torch.cat(all_trues).numpy()
    all_dates = torch.cat(all_dates).numpy()
    ap = average_precision_score(all_trues,all_preds)
    all_preds_bin = (all_preds > 0.5).astype(int)
    # Précision classique
    prec1 = precision_score(all_trues, all_preds_bin)

    prec_at = {}
    for thr in thresholds:
        mask = all_preds > thr
        if mask.sum() > 0:
            prec = all_trues[mask].sum() / mask.sum()
        else:
            prec = float('nan')
        prec_at[f'Prec@{thr}'] = prec
        prec_at[f'Num@{thr}'] = mask.sum() / len(all_preds)
    well_predicted_dates = all_dates[((all_preds > 0.5) == (all_trues == 1)).flatten()]
    badly_predicted_dates = all_dates[((all_preds > 0.5) != (all_trues == 1)).flatten()]

    return ap, avg_loss,prec1, prec_at,well_predicted_dates, badly_predicted_dates




import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_score

def train(train_loader,
          memory, gnn, win_pred, full_data,
          train_loader_ngh, eval_loader_ngh,
          optimizer, device, assoc, train_data, alpha,
          warmup_events: int = 5000):

    # Mode train
    memory.train()
    gnn.train()
    win_pred.train()
    memory.reset_state()
    train_loader_ngh.reset_state()
    eval_loader_ngh.reset_state()

    total_loss = 0.0
    num_events_seen = 0
    trained_events = 0

    all_preds, all_trues = [], []

    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        batch = batch.to(device)
        num_events_seen += batch.num_events

        # 1) Construire le sous-graphe
        nodes = torch.cat([batch.src, batch.dst]).unique()
        n_id, edge_index, e_id = train_loader_ngh(nodes)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # 2) Forward mémoire + GNN
        z, last_update = memory(n_id)
        t_e = full_data.t[e_id].to(device)
        msg_e = full_data.msg[e_id].to(device)
        h = gnn(z, last_update, edge_index, t_e, msg_e)
        match_feats =batch.msg
        # 3) Prédictions
        logit_win, margin, elo = win_pred(
            h[assoc[batch.src]], 
            h[assoc[batch.dst]],
            match_feats
        )
        y_win      = batch.y.view(-1, 1)
        y_setdiff  = batch.set_diff.view(-1, 1)
        y_elo_gain = batch.elo_gain.view(-1, 1)

        # 4) Warmup : on remplit juste la mémoire sans backprop
        if num_events_seen <= warmup_events:
            msg_with_label = torch.cat([batch.msg, y_win], dim=-1)
            memory.update_state(batch.src, batch.dst, batch.t, msg_with_label)
            train_loader_ngh.insert(batch.src, batch.dst)
            eval_loader_ngh.insert(batch.src, batch.dst)
            memory.detach()
            continue

        # 5) Calcul des pertes
        loss_win     = F.binary_cross_entropy_with_logits(logit_win, y_win)
        loss_setdiff = F.mse_loss(margin,     y_setdiff)
        loss_elo     = F.mse_loss(elo,        y_elo_gain)
        loss = alpha[0]*loss_win + alpha[1]*loss_setdiff + alpha[2]*loss_elo

        # 6) Backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7) Mise à jour mémoire APRÈS backward/step
        msg_with_label = torch.cat([batch.msg, y_win], dim=-1)
        memory.update_state(batch.src, batch.dst, batch.t, msg_with_label)
        train_loader_ngh.insert(batch.src, batch.dst)
        eval_loader_ngh.insert(batch.src, batch.dst)
        memory.detach()

        # 8) Collecte métriques
        all_preds.append(logit_win.sigmoid().detach().cpu())
        all_trues.append(y_win.detach().cpu())

        # 9) Comptage de la perte
        total_loss   += loss.item() * batch.num_events
        trained_events += batch.num_events

    # 10) Si jamais aucun événement n’a été entraîné
    if trained_events == 0:
        return 0.0, 0.0, 0.0

    # 11) Calcul final des métriques
    preds = torch.cat(all_preds).numpy()
    trues = torch.cat(all_trues).numpy()
    ap   = average_precision_score(trues, preds)
    prec = precision_score(trues, preds > 0.5)

    # 12) Perte moyenne sur les événements entraînés
    mean_loss = total_loss / trained_events

    return mean_loss, ap, prec

def train_debug(train_loader,
          memory, gnn, win_pred, full_data,
          train_loader_ngh, eval_loader_ngh,
          optimizer, device, assoc, train_data, alpha,lambda_recon=0.1,
          warmup_events: int = 5000):

    # Mode train
    memory.train()
    gnn.train()
    win_pred.train()
    memory.reset_state()
    train_loader_ngh.reset_state()
    eval_loader_ngh.reset_state()

    total_loss = 0.0
    num_events_seen = 0
    trained_events = 0

    all_preds, all_trues = [], []

    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        batch = batch.to(device)
        num_events_seen += batch.num_events

        # 1) Construire le sous-graphe
        nodes = torch.cat([batch.src, batch.dst]).unique()
        n_id1, edge_index1, e_id1 = train_loader_ngh(nodes)
        n_id, edge_index, e_id = train_loader_ngh(n_id1)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        # 2) Forward mémoire + GNN
        z, last_update = memory(n_id)
        t_e = full_data.t[e_id].to(device)
        msg_e = full_data.msg[e_id].to(device)
        h = gnn(z, last_update, edge_index, t_e, msg_e)
        match_feats =batch.msg
        # 3) Prédictions
        logit_win, margin, elo = win_pred(
            h[assoc[batch.src]], 
            h[assoc[batch.dst]],
            match_feats
        )
        y_win      = batch.y.view(-1, 1)
        y_setdiff  = batch.set_diff.view(-1, 1)
        y_elo_gain = batch.elo_gain.view(-1, 1)

        # 4) Warmup : on remplit juste la mémoire sans backprop
        if num_events_seen <= warmup_events:
            msg_with_label = torch.cat([batch.msg, y_win], dim=-1)
            memory.update_state(batch.src, batch.dst, batch.t, msg_with_label)
            train_loader_ngh.insert(batch.src, batch.dst)
            eval_loader_ngh.insert(batch.src, batch.dst)
            memory.detach()
            continue
        new_z, _ = memory(n_id)
        loss_recon = F.mse_loss(z.detach(), new_z)
        # 5) Calcul des pertes
        loss_win     = F.binary_cross_entropy_with_logits(logit_win, y_win)
        loss_setdiff = F.mse_loss(margin,     y_setdiff)
        loss_elo     = F.mse_loss(elo,        y_elo_gain)
        loss = alpha[0]*loss_win + alpha[1]*loss_setdiff + alpha[2]*loss_elo + lambda_recon * loss_recon

        # 6) Backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7) Mise à jour mémoire APRÈS backward/step
        msg_with_label = torch.cat([batch.msg, y_win], dim=-1)
        memory.update_state(batch.src, batch.dst, batch.t, msg_with_label)
        train_loader_ngh.insert(batch.src, batch.dst)
        eval_loader_ngh.insert(batch.src, batch.dst)
        memory.detach()

        # 8) Collecte métriques
        all_preds.append(logit_win.sigmoid().detach().cpu())
        all_trues.append(y_win.detach().cpu())

        # 9) Comptage de la perte
        total_loss   += loss.item() * batch.num_events
        trained_events += batch.num_events

    # 10) Si jamais aucun événement n’a été entraîné
    if trained_events == 0:
        return 0.0, 0.0, 0.0

    # 11) Calcul final des métriques
    preds = torch.cat(all_preds).numpy()
    trues = torch.cat(all_trues).numpy()
    ap   = average_precision_score(trues, preds)
    prec = precision_score(trues, preds > 0.5)

    # 12) Perte moyenne sur les événements entraînés
    mean_loss = total_loss / trained_events

    return mean_loss, ap, prec
def train_debug2(train_loader,
          memory, gnn, win_pred, full_data,
          train_loader_ngh, eval_loader_ngh,
          optimizer, device, assoc, train_data, alpha,lambda_recon=0.1,
          warmup_events: int = 5000):

    # Mode train
    memory.train()
    gnn.train()
    win_pred.train()
    memory.reset_state()
    train_loader_ngh.reset_state()
    eval_loader_ngh.reset_state()

    total_loss = 0.0
    num_events_seen = 0
    trained_events = 0

    all_preds, all_trues = [], []

    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        batch = batch.to(device)
        num_events_seen += batch.num_events

        # 1) Construire le sous-graphe
        nodes = torch.cat([batch.src, batch.dst]).unique()
        n_id1, edge_index1, e_id1 = train_loader_ngh(nodes)
        n_id, edge_index, e_id = train_loader_ngh(n_id1)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        # 2) Forward mémoire + GNN
        z, last_update = memory(n_id)
        t_e = full_data.t[e_id].to(device)
        msg_e = full_data.msg[e_id].to(device)
        h = gnn(z, last_update, edge_index, t_e, msg_e)
        match_feats =batch.msg

        # 3) Prédictions
        logit_win = win_pred(
            h[assoc[batch.src]], 
            h[assoc[batch.dst]],
            match_feats
        )
        y_win      = batch.y.view(-1, 1)
        y_closeness      = batch.closeness.view(-1, 1)
  

        # 4) Warmup : on remplit juste la mémoire sans backprop
        if num_events_seen <= warmup_events:
            msg_with_label = torch.cat([batch.msg], dim=-1)
            memory.update_state(batch.src, batch.dst, batch.t, msg_with_label)
            train_loader_ngh.insert(batch.src, batch.dst)
            eval_loader_ngh.insert(batch.src, batch.dst)
            memory.detach()
            continue
        new_z, _ = memory(n_id)
        loss_recon = F.mse_loss(z.detach(), new_z)
        # 5) Calcul des pertes
        loss_win     = F.binary_cross_entropy_with_logits(logit_win, y_win)

        loss = loss_win 

        # 6) Backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7) Mise à jour mémoire APRÈS backward/step
        msg_with_label = torch.cat([batch.msg], dim=-1)
        memory.update_state(batch.src, batch.dst, batch.t, msg_with_label)
        train_loader_ngh.insert(batch.src, batch.dst)
        eval_loader_ngh.insert(batch.src, batch.dst)
        memory.detach()

        # 8) Collecte métriques
        all_preds.append(logit_win.sigmoid().detach().cpu())
        all_trues.append(y_win.detach().cpu())

        # 9) Comptage de la perte
        total_loss   += loss.item() * batch.num_events
        trained_events += batch.num_events

    # 10) Si jamais aucun événement n’a été entraîné
    if trained_events == 0:
        return 0.0, 0.0, 0.0

    # 11) Calcul final des métriques
    preds = torch.cat(all_preds).numpy()
    trues = torch.cat(all_trues).numpy()
    ap   = average_precision_score(trues, preds)
    prec = precision_score(trues, preds > 0.5)

    # 12) Perte moyenne sur les événements entraînés
    mean_loss = total_loss / trained_events

    return mean_loss, ap, prec




def compute_alpha(epoch, num_epochs):
    if epoch < (num_epochs/2):
        progress = epoch / (num_epochs/2)
        alpha_win = 0.1 + 0.8 * progress
        alpha_reg = (1.0 - alpha_win) / 2
    else:
        alpha_win = 0.9
        alpha_reg = 0.05

    return [alpha_win, alpha_reg, alpha_reg]

