#!/usr/bin/env python3
"""
experiments/benchmark/li2018/train.py

Train Li2018 baseline with two-stage training (Li et al. ISMIR 2018, Section 3.3):
- Stage 1 (30 epochs): Joint constraint only (L1 loss, Eq. 1)
- Stage 2 (120 epochs): Joint + Limb constraint (Eq. 2)
- Total: 150 epochs (fixed schedule, no early stopping in original paper)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.benchmark.li2018.model import Li2018Model, compute_li2018_loss
from experiments.benchmark.li2018.dataset import get_li2018_train_loader, get_li2018_val_loader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def train_epoch(model, train_loader, optimizer, device, stage, limb_weight):
    model.train()
    total_loss, total_joint, total_limb = 0, 0, 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Training ({stage})")
    for batch in pbar:
        joints = batch['joints'].to(device)
        midi = {k: v.to(device) for k, v in batch['midi_features'].items()}
        metric = batch['metric_matrix'].to(device)
        
        optimizer.zero_grad()
        
        pred = model(midi, metric)
        losses = compute_li2018_loss(pred, joints, stage=stage, limb_weight=limb_weight)
        
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += losses['loss'].item()
        total_joint += losses['joint_loss'].item()
        total_limb += losses['limb_loss'].item()
        num_batches += 1
        
        pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")
    
    return {
        'loss': total_loss / num_batches,
        'joint_loss': total_joint / num_batches,
        'limb_loss': total_limb / num_batches,
    }


@torch.no_grad()
def validate(model, val_loader, device, stage, limb_weight):
    model.eval()
    total_loss, total_joint, total_limb = 0, 0, 0
    num_batches = 0
    
    for batch in val_loader:
        joints = batch['joints'].to(device)
        midi = {k: v.to(device) for k, v in batch['midi_features'].items()}
        metric = batch['metric_matrix'].to(device)
        
        pred = model(midi, metric)
        losses = compute_li2018_loss(pred, joints, stage=stage, limb_weight=limb_weight)
        
        total_loss += losses['loss'].item()
        total_joint += losses['joint_loss'].item()
        total_limb += losses['limb_loss'].item()
        num_batches += 1
    
    return {
        'val_loss': total_loss / num_batches,
        'val_joint_loss': total_joint / num_batches,
        'val_limb_loss': total_limb / num_batches,
    }


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    train_loader = get_li2018_train_loader(
        args.cache_dir, args.batch_size, args.window_size, args.num_workers
    )
    val_loader = get_li2018_val_loader(
        args.cache_dir, args.batch_size, args.window_size, args.num_workers
    )
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # Model
    model = Li2018Model(
        window_size=args.cnn_window_size,
        piano_feat_dim=args.piano_feat_dim,
        metric_feat_dim=args.metric_feat_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        use_metric=args.use_metric,
        dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    if args.use_wandb and HAS_WANDB:
        wandb.init(
            project="pianoid-benchmark",
            name=f"li2018-{datetime.now().strftime('%m%d-%H%M')}",
            config=vars(args)
        )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Two-stage training
    total_epochs = args.stage1_epochs + args.stage2_epochs
    
    for epoch in range(total_epochs):
        # Determine stage
        if epoch < args.stage1_epochs:
            stage = 'joint'
            stage_name = "Stage 1 (Joint)"
        else:
            stage = 'limb'
            stage_name = "Stage 2 (Joint+Limb)"
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, stage, args.limb_weight
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, device, stage, args.limb_weight
        )
        
        scheduler.step(val_metrics['val_loss'])
        lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{total_epochs} [{stage_name}]")
        print(f"  Train Loss: {train_metrics['loss']:.4f} (joint: {train_metrics['joint_loss']:.4f}, limb: {train_metrics['limb_loss']:.4f})")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}, LR: {lr:.6f}")
        
        if args.use_wandb and HAS_WANDB:
            wandb.log({
                'epoch': epoch + 1,
                'stage': stage,
                **train_metrics,
                **val_metrics,
                'lr': lr,
            })
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'stage': stage,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': {
                    'window_size': args.cnn_window_size,
                    'piano_feat_dim': args.piano_feat_dim,
                    'metric_feat_dim': args.metric_feat_dim,
                    'lstm_hidden': args.lstm_hidden,
                    'lstm_layers': args.lstm_layers,
                    'use_metric': args.use_metric,
                    'dropout': args.dropout,
                },
            }, ckpt_dir / 'best_model.pt')
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping (only in stage 2)
        if stage == 'limb' and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'val_loss': val_metrics['val_loss'],
        'config': {
            'window_size': args.cnn_window_size,
            'piano_feat_dim': args.piano_feat_dim,
            'metric_feat_dim': args.metric_feat_dim,
            'lstm_hidden': args.lstm_hidden,
            'lstm_layers': args.lstm_layers,
            'use_metric': args.use_metric,
            'dropout': args.dropout,
        },
    }, ckpt_dir / 'final_model.pt')
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    
    if args.use_wandb and HAS_WANDB:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Li2018 baseline")
    
    # Data
    parser.add_argument('--cache_dir', default='MOSA_dataset/training_cache')
    parser.add_argument('--window_size', type=int, default=480, help='Training window size')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--cnn_window_size', type=int, default=120, help='CNN sliding window (±1sec at 60fps)')
    parser.add_argument('--piano_feat_dim', type=int, default=50)
    parser.add_argument('--metric_feat_dim', type=int, default=10)
    parser.add_argument('--lstm_hidden', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--use_metric', action='store_true', default=True)
    parser.add_argument('--no_metric', dest='use_metric', action='store_false')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training (Two-stage warmup from Li et al. 2018 Section 3.3, extended to unified 500-epoch budget)
    parser.add_argument('--stage1_epochs', type=int, default=30, help='Epochs for Stage 1 joint-only warmup (paper: 30)')
    parser.add_argument('--stage2_epochs', type=int, default=470, help='Epochs for Stage 2 joint+limb (extended from paper 120 so stage1+stage2=500 unified budget)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='β = 10^-8 in original paper')
    parser.add_argument('--limb_weight', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (unified 50 epochs across all baselines)')
    
    # Output
    parser.add_argument('--checkpoint_dir', default='experiments/benchmark/li2018/checkpoints')
    parser.add_argument('--use_wandb', action='store_true')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
