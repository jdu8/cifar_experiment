
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import umap


def fit_umap_reducer(train_embs, max_points=5000, seed=42):
    N   = train_embs.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(N, min(N, max_points), replace=False))
    print(f"Fitting UMAP on {len(idx)} points (once)...")
    reducer = umap.UMAP(n_components=2, n_jobs=-1)
    reducer.fit(train_embs[idx].astype(np.float32))
    print("UMAP fitted.")
    return reducer


def save_epoch_plots(reducer, train_embs, labels, losses, weights,
                     out_dir, epoch, n_classes=10):
    """
    Save a single 3-panel PNG per epoch: colored by class, loss, weight.
    Returns the saved path.
    """
    projected = reducer.transform(train_embs.astype(np.float32))
    s         = max(1, 8000 // len(labels))   # point size scales with N

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Epoch {epoch}", fontsize=13)

    # ── Class ──────────────────────────────────────────────────────────
    cmap_class = 'tab10' if n_classes <= 10 else 'nipy_spectral'
    sc0 = axes[0].scatter(projected[:, 0], projected[:, 1],
                          c=labels.astype(int), cmap=cmap_class,
                          s=s, alpha=0.5, vmin=0, vmax=n_classes - 1)
    fig.colorbar(sc0, ax=axes[0], fraction=0.03)
    axes[0].set_title('class')
    axes[0].axis('off')

    # ── Loss ───────────────────────────────────────────────────────────
    sc1 = axes[1].scatter(projected[:, 0], projected[:, 1],
                          c=losses, cmap='coolwarm', s=s, alpha=0.5,
                          vmin=np.percentile(losses, 5),
                          vmax=np.percentile(losses, 95))
    fig.colorbar(sc1, ax=axes[1], fraction=0.03, label='loss')
    axes[1].set_title('loss')
    axes[1].axis('off')

    # ── Weight ─────────────────────────────────────────────────────────
    sc2 = axes[2].scatter(projected[:, 0], projected[:, 1],
                          c=weights, cmap='RdYlGn', s=s, alpha=0.5,
                          vmin=weights.min(), vmax=weights.max())
    fig.colorbar(sc2, ax=axes[2], fraction=0.03, label='weight')
    axes[2].set_title('weight')
    axes[2].axis('off')

    path = os.path.join(out_dir, f'proj_epoch{epoch:03d}.png')
    plt.tight_layout()
    plt.savefig(path, dpi=80, bbox_inches='tight')
    plt.close()
    return path


def make_displacement_plot(displacements_per_epoch, train_labels,
                           losses_per_epoch, out_path, n_classes=10, classes=None):
    n_epochs    = len(displacements_per_epoch)
    epoch_range = list(range(2, n_epochs + 2))
    final_losses = losses_per_epoch[-1]
    hard_mask    = final_losses >= 1.0
    easy_mask    = final_losses <  0.5

    show_per_class = (n_classes <= 20) and (classes is not None)
    ncols = 2 if show_per_class else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6))
    if not show_per_class:
        axes = [axes]
    fig.suptitle("Embedding Space Displacement Across Training", fontsize=13)

    mean_all  = [d.mean() for d in displacements_per_epoch]
    mean_hard = [d[hard_mask].mean() if hard_mask.sum() > 0 else 0.0
                 for d in displacements_per_epoch]
    mean_easy = [d[easy_mask].mean() if easy_mask.sum() > 0 else 0.0
                 for d in displacements_per_epoch]

    axes[0].plot(epoch_range, mean_all,  label='All',  color='steelblue', linewidth=2)
    axes[0].plot(epoch_range, mean_hard, label='Hard', color='tomato',    linewidth=2)
    axes[0].plot(epoch_range, mean_easy, label='Easy', color='green',     linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean L2 displacement')
    axes[0].set_title('Displacement: Hard vs Easy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if show_per_class:
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_classes, 20)))
        for cls in range(n_classes):
            mask    = train_labels == cls
            label   = classes[cls] if classes else str(cls)
            mean_cls = [d[mask].mean() if mask.sum() > 0 else 0.0
                        for d in displacements_per_epoch]
            axes[1].plot(epoch_range, mean_cls, label=label,
                         color=colors[cls % 20], linewidth=1.2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean L2 displacement')
        axes[1].set_title('Displacement: Per Class')
        axes[1].legend(fontsize=6, ncol=3)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Displacement plot saved to {out_path}")
