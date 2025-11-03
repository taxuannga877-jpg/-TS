"""
过渡态预测的训练脚本

使用方法：
    python train.py --config config.yaml
    python train.py --config config.yaml --resume outputs/run_xxx/checkpoint.pt
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from datetime import datetime

# 添加模型和数据路径
sys.path.insert(0, os.path.dirname(__file__))

# 从完整文件导入（因为还未完全模块化）
from models.ts_predictor_complete import create_reaction_centric_model, ReactionCentricLoss
from data.dataset_complete import TSFFHybridDataset, collate_fn
from utils.metrics import calculate_rmsd


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练TS预测模型')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='要恢复的检查点路径')
    parser.add_argument('--train_dir', type=str, default=None,
                       help='覆盖训练数据目录')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='覆盖输出目录')
    parser.add_argument('--device', type=str, default=None,
                       help='设备：cuda 或 cpu')
    
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_rmsd = 0
    count = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        data_r = batch['data_r'].to(device)
        data_p = batch['data_p'].to(device)
        data_ts = batch['data_ts'].to(device)
        
        # 混合精度前向传播
        with autocast(enabled=config['training']['mixed_precision']):
            ts_pred, alpha, delta, confidence, reaction_scores = model(data_r, data_p)
            loss, loss_dict = criterion(
                ts_pred, data_ts.pos, alpha, confidence, reaction_scores,
                data_r, data_p, data_ts.batch
            )
        
        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # 计算RMSD
        with torch.no_grad():
            batch_rmsd = 0
            batch_size = data_ts.batch.max().item() + 1
            for i in range(batch_size):
                mask = (data_ts.batch == i)
                pred = ts_pred[mask].cpu().numpy()
                true = data_ts.pos[mask].cpu().numpy()
                batch_rmsd += calculate_rmsd(pred, true)
            batch_rmsd /= batch_size
        
        total_loss += loss.item()
        total_rmsd += batch_rmsd
        count += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rmsd': f'{batch_rmsd:.4f}',
            'mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
        })
    
    return {
        'loss': total_loss / count,
        'rmsd': total_rmsd / count
    }


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    total_rmsd = 0
    success_count = 0
    total_count = 0
    count = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Validate]')
        for batch in pbar:
            data_r = batch['data_r'].to(device)
            data_p = batch['data_p'].to(device)
            data_ts = batch['data_ts'].to(device)
            
            ts_pred, alpha, delta, confidence, reaction_scores = model(data_r, data_p)
            loss, loss_dict = criterion(
                ts_pred, data_ts.pos, alpha, confidence, reaction_scores,
                data_r, data_p, data_ts.batch
            )
            
            # 计算RMSD
            batch_rmsd = 0
            batch_size = data_ts.batch.max().item() + 1
            for i in range(batch_size):
                mask = (data_ts.batch == i)
                pred = ts_pred[mask].cpu().numpy()
                true = data_ts.pos[mask].cpu().numpy()
                rmsd = calculate_rmsd(pred, true)
                
                batch_rmsd += rmsd
                total_count += 1
                if rmsd < 0.5:  # 成功阈值
                    success_count += 1
            
            batch_rmsd /= batch_size
            
            total_loss += loss.item()
            total_rmsd += batch_rmsd
            count += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'rmsd': f'{batch_rmsd:.4f}'})
    
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    
    return {
        'loss': total_loss / count,
        'rmsd': total_rmsd / count,
        'success_rate': success_rate
    }


def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
    if args.train_dir:
        config['data']['train_path'] = args.train_dir
    if args.output_dir:
        config['logging']['output_dir'] = args.output_dir
    if args.device:
        config['device'] = args.device
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cudnn.benchmark = True
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['logging']['output_dir'], f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Setup device
    device = torch.device(config['device'])
    
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "TS Prediction Model Training" + " "*30 + "║")
    print("╚" + "="*78 + "╝\n")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Output: {output_dir}")
    print(f"Mixed Precision: {'Enabled' if config['training']['mixed_precision'] else 'Disabled'}")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    dataset = TSFFHybridDataset(
        data_dir=config['data']['train_path'],
        cutoff=config['model']['cutoff'],
        augment=config['data']['augment']
    )
    
    val_size = int(len(dataset) * config['data']['val_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = create_reaction_centric_model(config['model']).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)\n")
    
    # Loss and optimizer
    criterion = ReactionCentricLoss(
        coord_weight=config['loss']['coord_weight'],
        dist_weight=config['loss']['distance_weight'],
        center_weight=config['loss']['center_weight'],
        confidence_weight=config['loss']['confidence_weight'],
        collision_weight=config['loss']['collision_weight']
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # LR scheduler
    def lr_lambda(epoch):
        if epoch < config['training']['warmup_epochs']:
            return (epoch + 1) / config['training']['warmup_epochs']
        progress = (epoch - config['training']['warmup_epochs']) / \
                   (config['training']['epochs'] - config['training']['warmup_epochs'])
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print("Starting training...\n")
    best_val_rmsd = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, config)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('RMSD/train', train_metrics['rmsd'], epoch)
        writer.add_scalar('RMSD/val', val_metrics['rmsd'], epoch)
        writer.add_scalar('SuccessRate/val', val_metrics['success_rate'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*80}")
        print(f"Train: Loss={train_metrics['loss']:.4f}, RMSD={train_metrics['rmsd']:.4f} Å")
        print(f"Val:   Loss={val_metrics['loss']:.4f}, RMSD={val_metrics['rmsd']:.4f} Å, Success={val_metrics['success_rate']:.2f}%")
        print(f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_metrics['rmsd'] < best_val_rmsd:
            best_val_rmsd = val_metrics['rmsd']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_rmsd': val_metrics['rmsd'],
                'success_rate': val_metrics['success_rate'],
                'config': config
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pt'))
            print(f"✓ Saved best model (Val RMSD: {val_metrics['rmsd']:.4f} Å)")
        
        # Periodic save
        if epoch % config['logging']['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_rmsd': val_metrics['rmsd'],
                'config': config
            }
            torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    writer.close()
    
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"{'='*80}")
    print(f"Best validation RMSD: {best_val_rmsd:.4f} Å")
    print(f"Model saved in: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

