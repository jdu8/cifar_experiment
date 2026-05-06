
import numpy as np
import torch
import faiss
from scipy.stats import spearmanr


def _normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / norms).astype(np.float32)


def _build_faiss_index(train_embs_norm):
    """
    Build a flat GPU index for exact kNN search.
    Returns index on GPU if available, CPU otherwise.
    """
    d = train_embs_norm.shape[1]

    try:
        res   = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, d)  # inner product on normalized = cosine
        index.add(train_embs_norm)
        return index, True
    except Exception:
        # Fall back to CPU
        index = faiss.IndexFlatIP(d)
        index.add(train_embs_norm)
        return index, False


def compute_neighbors_faiss(val_embs, train_embs, k=20):
    """
    Find k nearest train neighbors for each val point using faiss.
    Returns distances (N, k) and indices (N, k).
    """
    train_norm = _normalize(train_embs)
    val_norm   = _normalize(val_embs)

    index, on_gpu = _build_faiss_index(train_norm)
    distances, indices = index.search(val_norm, k)

    return distances, indices


def compute_within_class_neighbors_faiss(val_embs, val_labels,
                                         train_embs, train_labels, k=20):
    """
    For each val point find k nearest train neighbors
    restricted to same class only, using faiss.
    Returns indices (N, k) into global train array.
    """
    val_norm   = _normalize(val_embs)
    train_norm = _normalize(train_embs)

    n_val      = len(val_labels)
    indices_within = np.full((n_val, k), -1, dtype=np.int64)

    for cls in range(10):
        val_mask   = val_labels   == cls
        train_mask = train_labels == cls

        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        val_cls   = val_norm[val_mask]
        train_cls = train_norm[train_mask]

        k_cls = min(k, train_cls.shape[0] - 1)

        try:
            res   = faiss.StandardGpuResources()
            d     = train_cls.shape[1]
            index = faiss.GpuIndexFlatIP(res, d)
            index.add(train_cls)
        except Exception:
            d     = train_cls.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(train_cls)

        _, idx_cls = index.search(val_cls, k_cls)

        # Map back to global train indices
        train_cls_global = np.where(train_mask)[0]
        val_positions    = np.where(val_mask)[0]

        for i, val_pos in enumerate(val_positions):
            global_idx = train_cls_global[idx_cls[i]]
            indices_within[val_pos, :k_cls] = global_idx

    return indices_within


def evaluation_pass(loader, model, criterion, device):
    """
    Frozen weights pass — returns per-sample losses, embeddings,
    labels, accuracy.
    """
    model.eval()
    all_losses, all_embs, all_labels = [], [], []
    correct = total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, emb    = model(inputs, return_embedding=True)
            loss           = criterion(logits, labels)

            all_losses.append(loss.cpu().numpy())
            all_embs.append(emb.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)

    losses = np.concatenate(all_losses)
    embs   = np.concatenate(all_embs)
    labels = np.concatenate(all_labels)
    acc    = correct / total

    return losses, embs, labels, acc


def compute_purity(val_labels, train_labels, neighbor_indices, k=20):
    """
    For each val point compute fraction of neighbors that share its class.
    """
    purity = np.zeros(len(val_labels))
    for i, cls in enumerate(val_labels):
        nbr_idx = neighbor_indices[i]
        nbr_idx = nbr_idx[nbr_idx >= 0]
        if len(nbr_idx) == 0:
            continue
        purity[i] = (train_labels[nbr_idx] == cls).mean()
    return purity


def compute_spearman(val_losses, train_losses, neighbor_indices):
    """
    Spearman correlation between val loss and mean neighbor train loss.
    """
    mean_nb_loss = np.array([
        train_losses[neighbor_indices[i][neighbor_indices[i] >= 0]].mean()
        if (neighbor_indices[i] >= 0).any() else 0.0
        for i in range(len(val_losses))
    ])
    with np.errstate(invalid='ignore'):
        rho, pval = spearmanr(val_losses, mean_nb_loss)
    return (rho if not np.isnan(rho) else 0.0), pval


def compute_displacement(embs_prev, embs_curr):
    """
    Per-point L2 displacement between consecutive epoch embeddings.
    """
    return np.linalg.norm(embs_curr - embs_prev, axis=1)


def compute_metrics_summary(val_losses, val_labels, val_embs,
                             train_losses, train_labels, train_embs,
                             prev_train_embs=None, k=20):
    """
    Compute all metrics in one call using faiss for fast kNN.
    Returns flat dict suitable for wandb.log()
    """
    # Global neighbors
    _, global_indices = compute_neighbors_faiss(
        val_embs, train_embs, k=k)

    # Purity
    purity = compute_purity(
        val_labels, train_labels, global_indices, k=k)

    # Global spearman
    rho_global, _ = compute_spearman(
        val_losses, train_losses, global_indices)

    # Within class neighbors
    within_indices = compute_within_class_neighbors_faiss(
        val_embs, val_labels, train_embs, train_labels, k=k)

    # Within class spearman
    rho_within, _ = compute_spearman(
        val_losses, train_losses, within_indices)

    # Stratified within class rho
    bins = [
        ('easy',     val_losses <  0.5),
        ('moderate', (val_losses >= 0.5) & (val_losses < 1.0)),
        ('hard',     (val_losses >= 1.0) & (val_losses < 2.0)),
        ('vhard',    (val_losses >= 2.0) & (val_losses < 3.0)),
        ('extreme',   val_losses >= 3.0),
    ]

    strat_rhos = {}
    for name, mask in bins:
        if mask.sum() < 10:
            strat_rhos[f'rho_within_{name}'] = float('nan')
            continue
        subset_indices = np.where(mask)[0]
        mean_nb = np.array([
            train_losses[within_indices[i][within_indices[i] >= 0]].mean()
            if (within_indices[i] >= 0).any() else 0.0
            for i in subset_indices
        ])
        with np.errstate(invalid='ignore'):
            r, _ = spearmanr(val_losses[mask], mean_nb)
        strat_rhos[f'rho_within_{name}'] = float(r) if not np.isnan(r) else 0.0

    # Displacement
    disp_stats = {}
    if prev_train_embs is not None:
        disp = compute_displacement(prev_train_embs, train_embs)
        disp_stats = {
            'displacement_mean': float(disp.mean()),
            'displacement_std':  float(disp.std()),
            'displacement_max':  float(disp.max()),
        }
    else:
        disp_stats = {
            'displacement_mean': float('nan'),
            'displacement_std':  float('nan'),
            'displacement_max':  float('nan'),
        }

    metrics = {
        'rho_global':    rho_global,
        'rho_within':    rho_within,
        'purity_mean':   float(purity.mean()),
        'purity_median': float(np.median(purity)),
        'n_displaced':   int((purity < 0.15).sum()),
        'val_loss_mean': float(val_losses.mean()),
        'val_loss_std':  float(val_losses.std()),
        'n_hard':        int((val_losses >= 1.0).sum()),
        'n_easy':        int((val_losses <  0.5).sum()),
        **strat_rhos,
        **disp_stats,
    }

    return metrics, global_indices, within_indices, purity
