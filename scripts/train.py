# scripts/train.py

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data      import get_dataloaders
from utils.model     import get_model
from utils.metrics   import (evaluation_pass, compute_metrics_summary,
                              compute_displacement)
from utils.animation import (fit_umap_reducer, project_and_save,
                              make_animation_from_projections,
                              make_displacement_plot)

CLASSES = ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir',      type=str,   required=True)
    p.add_argument('--data_dir',     type=str,   default='/content/data')
    p.add_argument('--model',        type=str,   default='resnet18')
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=256)
    p.add_argument('--lr',           type=float, default=0.1)
    p.add_argument('--seed',         type=int,   default=0)
    p.add_argument('--k',            type=int,   default=20)
    p.add_argument('--val_size',     type=int,   default=5000)
    p.add_argument('--patience',     type=int,   default=10)
    p.add_argument('--smoke_test',   action='store_true')
    p.add_argument('--wandb_project',type=str,   default='cifar')
    p.add_argument('--run_name',     type=str,   default=None)
    p.add_argument('--sample_every', type=int,   default=1)
    p.add_argument('--no_animation', action='store_true')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Wandb
    run_name = args.run_name or f"baseline_{args.model}_s{args.seed}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        dir=args.out_dir
    )

    # Data
    train_loader, _, train_eval_loader, val_loader, test_loader, split_info = \
        get_dataloaders(
            data_dir=args.data_dir,
            val_size=args.val_size,
            batch_size=args.batch_size,
            seed=args.seed,
            smoke_test=args.smoke_test
        )

    n_train = split_info['n_train']

    # Model
    model     = get_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Storage for animation
    train_embs_history   = []
    train_losses_history = []
    train_labels_arr     = None
    displacements        = []
    best_test_acc        = 0.0
    patience_counter     = 0

    for epoch in range(args.epochs):

        # ── Training pass ──────────────────────────────────────────────────
        model.train()
        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch+1:03d}/{args.epochs} [Train]",
                         leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss   = criterion(logits, labels).mean()
            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()

        # ── Evaluation passes ──────────────────────────────────────────────
        train_losses, train_embs, train_labels, train_acc = \
            evaluation_pass(train_eval_loader, model, criterion, device)
        val_losses,   val_embs,   val_labels,   val_acc   = \
            evaluation_pass(val_loader,        model, criterion, device)
        test_losses,  test_embs,  test_labels,  test_acc  = \
            evaluation_pass(test_loader,       model, criterion, device)

        # Store for animation
        if train_labels_arr is None:
            train_labels_arr = train_labels
        # Save previous embeddings before updating history
        prev_train_embs = train_embs_history[-1] if train_embs_history else None
        # Displacement — only need current and previous
        if prev_train_embs is not None:
            disp = compute_displacement(prev_train_embs, train_embs)
            displacements.append(disp)

        # Keep only last epoch in memory
        train_embs_history = [train_embs]
        train_losses_history.append(train_losses)

        # Fit UMAP once on epoch 1, transform each epoch
        if not args.no_animation:
            if epoch == 0:
                umap_reducer = fit_umap_reducer(train_embs, max_points=5000)
            project_and_save(umap_reducer, train_embs,
                             args.out_dir, epoch + 1)

        # ── Metrics ────────────────────────────────────────────────────────

        metrics, _, _, _ = compute_metrics_summary(
            val_losses, val_labels, val_embs,
            train_losses, train_labels, train_embs,
            prev_train_embs=prev_train_embs,
            k=args.k
        )

        # Per class test accuracy
        per_class_acc = {}
        for cls in range(10):
            mask = test_labels == cls
            if mask.sum() > 0:
                per_class_acc[f'test_acc_{CLASSES[cls]}'] = \
                    (test_losses[mask] < np.log(2)).mean()

        # ── Wandb log ──────────────────────────────────────────────────────
        wandb.log({
            'epoch':          epoch + 1,
            'train_loss':     train_losses.mean(),
            'val_loss':       val_losses.mean(),
            'test_loss':      test_losses.mean(),
            'train_acc':      train_acc,
            'val_acc':        val_acc,
            'test_acc':       test_acc,
            'lr':             scheduler.get_last_lr()[0],
            **metrics,
            **per_class_acc,
        })

        print(f"Epoch {epoch+1:03d} | "
              f"Train: {train_acc:.3f} | "
              f"Val: {val_acc:.3f} | "
              f"Test: {test_acc:.3f} | "
              f"ρ_within: {metrics['rho_within']:.3f}")

        # ── Patience on test acc ───────────────────────────────────────────
        if test_acc > best_test_acc:
            best_test_acc    = test_acc
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, 'best_model.pt'))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(best test acc: {best_test_acc:.3f})")
            break

    # ── Animations ─────────────────────────────────────────────────────────
    if not args.no_animation:
        print("Generating animations...")

        make_animation_from_projections(
            args.out_dir, train_labels_arr, train_losses_history,
            out_path=os.path.join(args.out_dir, 'embedding_class.gif'),
            classes=CLASSES, color_by='class',
        )

        make_animation_from_projections(
            args.out_dir, train_labels_arr, train_losses_history,
            out_path=os.path.join(args.out_dir, 'embedding_loss.gif'),
            classes=CLASSES, color_by='loss',
        )

        if displacements:
            make_displacement_plot(
                displacements, train_labels_arr, train_losses_history,
                out_path=os.path.join(args.out_dir, 'displacement.png'),
                classes=CLASSES
            )

        wandb.log({
            'embedding_class_animation':
                wandb.Video(os.path.join(args.out_dir, 'embedding_class.gif'), format='gif'),
            'embedding_loss_animation':
                wandb.Video(os.path.join(args.out_dir, 'embedding_loss.gif'), format='gif'),
        })

    wandb.log({'best_test_acc': best_test_acc})
    wandb.finish()
    print(f"Done. Best test acc: {best_test_acc:.3f}")


if __name__ == '__main__':
    main()