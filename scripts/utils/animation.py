
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import umap
import os
import glob


def fit_umap_reducer(train_embs_epoch1, max_points=5000, seed=42):
    """
    Fit UMAP once on a subsample of epoch 1 embeddings.
    Returns the fitted reducer to use for transform() in subsequent epochs.
    """
    N = train_embs_epoch1.shape[0]
    rng = np.random.RandomState(seed)

    if N > max_points:
        sample_idx = np.sort(rng.choice(N, max_points, replace=False))
    else:
        sample_idx = np.arange(N)

    print(f"Fitting UMAP on {len(sample_idx)} points (once)...")
    reducer = umap.UMAP(n_components=2, n_jobs=-1)
    reducer.fit(train_embs_epoch1[sample_idx].astype(np.float32))
    print("UMAP fitted.")
    return reducer


def project_and_save(reducer, train_embs, out_dir, epoch):
    """
    Project embeddings into fitted UMAP space and save as float16.
    Much cheaper than storing full 512d embeddings.
    """
    projected = reducer.transform(
        train_embs.astype(np.float32)).astype(np.float16)
    path = os.path.join(out_dir, f'proj_epoch{epoch:03d}.npy')
    np.save(path, projected)
    return projected


def make_animation_from_projections(out_dir, labels, losses_per_epoch,
                                     out_path, classes, color_by="class",
                                     fps=4):
    """
    Load saved 2D projections from disk and make animation.
    Called once after training completes.
    """
    proj_files = sorted(glob.glob(os.path.join(out_dir, 'proj_epoch*.npy')))
    if not proj_files:
        print("No projection files found, skipping animation.")
        return

    labels_int = labels.astype(int)
    n_classes  = len(classes)
    n_epochs   = len(proj_files)

    print(f"Making animation from {n_epochs} projection files...")

    fig, ax = plt.subplots(figsize=(8, 8))
    colorbar_added = False

    def update(frame):
        nonlocal colorbar_added
        ax.clear()
        proj = np.load(proj_files[frame]).astype(np.float32)
        ep   = int(os.path.basename(proj_files[frame])
                   .replace('proj_epoch','').replace('.npy',''))

        if color_by == "class":
            sc = ax.scatter(
                proj[:, 0], proj[:, 1],
                c=labels_int, cmap="tab10",
                s=2, alpha=0.6,
                vmin=0, vmax=n_classes - 1
            )
            if not colorbar_added:
                cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
                cbar.set_ticks(range(n_classes))
                cbar.set_ticklabels(classes, fontsize=7)
                colorbar_added = True
        else:
            losses = losses_per_epoch[frame]
            sc     = ax.scatter(
                proj[:, 0], proj[:, 1],
                c=losses, cmap="coolwarm",
                s=2, alpha=0.6,
                vmin=np.percentile(losses, 5),
                vmax=np.percentile(losses, 95)
            )
            if not colorbar_added:
                fig.colorbar(sc, ax=ax, fraction=0.03,
                             pad=0.04, label="loss")
                colorbar_added = True

        ax.set_title(f"Epoch {ep} | colored by {color_by}", fontsize=12)
        ax.axis("off")
        return sc,

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_epochs,
        interval=1000 // fps,
        blit=False
    )
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close()
    print(f"Animation saved to {out_path}")


def make_displacement_plot(displacements_per_epoch, train_labels,
                           losses_per_epoch, out_path, classes):
    n_epochs    = len(displacements_per_epoch)
    epoch_range = list(range(2, n_epochs + 2))

    final_losses = losses_per_epoch[-1]
    hard_mask    = final_losses >= 1.0
    easy_mask    = final_losses <  0.5

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Embedding Space Displacement Across Training", fontsize=13)

    mean_all  = [d.mean() for d in displacements_per_epoch]
    mean_hard = [d[hard_mask].mean() if hard_mask.sum() > 0
                 else 0.0 for d in displacements_per_epoch]
    mean_easy = [d[easy_mask].mean() if easy_mask.sum() > 0
                 else 0.0 for d in displacements_per_epoch]

    axes[0].plot(epoch_range, mean_all,
                 label="All",  color="steelblue", linewidth=2)
    axes[0].plot(epoch_range, mean_hard,
                 label="Hard", color="tomato",    linewidth=2)
    axes[0].plot(epoch_range, mean_easy,
                 label="Easy", color="green",     linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean L2 displacement from previous epoch")
    axes[0].set_title("Displacement: Hard vs Easy points")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for cls in range(len(classes)):
        cls_mask = train_labels == cls
        mean_cls = [d[cls_mask].mean() if cls_mask.sum() > 0
                    else 0.0 for d in displacements_per_epoch]
        axes[1].plot(epoch_range, mean_cls,
                     label=classes[cls],
                     color=colors[cls], linewidth=1.5)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean L2 displacement")
    axes[1].set_title("Displacement: Per Class")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Displacement plot saved to {out_path}")
