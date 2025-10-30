"""
使用XYZ格式数据训练过渡态预测模型
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from ts_prediction.data.xyz_dataset import XYZTransitionStateDataset, collate_fn
from ts_prediction.models import TSPredictor, create_model
from ts_prediction.utils.metrics import calculate_rmsd


def setup_device():
    """设置计算设备"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        # 优化CPU性能
        num_threads = os.cpu_count()
        torch.set_num_threads(num_threads)
        print(f"✓ CPU线程数: {num_threads}")
        return torch.device('cpu')
    
    try:
        # 测试GPU兼容性
        device = torch.device('cuda')
        test_tensor = torch.randn(10, 10, device=device)
        test_embedding = nn.Embedding(100, 128).to(device)
        _ = test_embedding(torch.randint(0, 100, (5,), device=device))
        del test_tensor, test_embedding
        torch.cuda.empty_cache()
        print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
        return device
    except Exception as e:
        print(f"⚠️  GPU不兼容 ({e})，回退到CPU")
        num_threads = os.cpu_count()
        torch.set_num_threads(num_threads)
        print(f"✓ CPU线程数: {num_threads}")
        return torch.device('cpu')


def calc_dist_matrix(pos):
    """计算距离矩阵"""
    # pos: (B, N, 3)
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, N, N, 3)
    dist = torch.norm(diff, dim=-1)  # (B, N, N)
    return dist


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_rmsd = 0
    count = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        atomic_numbers = batch['atomic_numbers'].to(device)
        r_pos = batch['r_pos'].to(device)
        p_pos = batch['p_pos'].to(device)
        ts_pos = batch['ts_pos'].to(device)
        mask = batch['mask'].to(device)
        
        # 计算距离矩阵
        r_dist = calc_dist_matrix(r_pos)  # (B, N, N)
        p_dist = calc_dist_matrix(p_pos)  # (B, N, N)
        
        # 前向传播（批量处理）
        batch_size = atomic_numbers.size(0)
        pred_ts_list = []
        
        for i in range(batch_size):
            n_atoms = mask[i].sum().item()
            pred_ts_single, _ = model(
                atomic_numbers[i, :n_atoms],
                r_pos[i, :n_atoms],
                p_pos[i, :n_atoms],
                r_dist[i, :n_atoms, :n_atoms],
                p_dist[i, :n_atoms, :n_atoms]
            )
            # Padding回原始大小
            pred_ts_padded = torch.zeros_like(r_pos[i])
            pred_ts_padded[:n_atoms] = pred_ts_single
            pred_ts_list.append(pred_ts_padded)
        
        pred_ts = torch.stack(pred_ts_list, dim=0)  # (B, N, 3)
        
        # 计算损失（只对有效原子）
        loss = criterion(pred_ts[mask], ts_pos[mask])
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 计算RMSD
        with torch.no_grad():
            batch_rmsd = 0
            for i in range(len(batch['num_atoms'])):
                n_atoms = batch['num_atoms'][i]
                pred = pred_ts[i, :n_atoms].cpu().numpy()
                true = ts_pos[i, :n_atoms].cpu().numpy()
                rmsd = calculate_rmsd(pred, true)
                batch_rmsd += rmsd
            batch_rmsd /= len(batch['num_atoms'])
        
        total_loss += loss.item()
        total_rmsd += batch_rmsd
        count += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'rmsd': f'{batch_rmsd:.4f}'})
    
    return total_loss / count, total_rmsd / count


def validate(model, dataloader, criterion, device, threshold=0.5):
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
            atomic_numbers = batch['atomic_numbers'].to(device)
            r_pos = batch['r_pos'].to(device)
            p_pos = batch['p_pos'].to(device)
            ts_pos = batch['ts_pos'].to(device)
            mask = batch['mask'].to(device)
            
            # 计算距离矩阵
            r_dist = calc_dist_matrix(r_pos)
            p_dist = calc_dist_matrix(p_pos)
            
            # 前向传播（批量处理）
            batch_size = atomic_numbers.size(0)
            pred_ts_list = []
            
            for i in range(batch_size):
                n_atoms = mask[i].sum().item()
                pred_ts_single, _ = model(
                    atomic_numbers[i, :n_atoms],
                    r_pos[i, :n_atoms],
                    p_pos[i, :n_atoms],
                    r_dist[i, :n_atoms, :n_atoms],
                    p_dist[i, :n_atoms, :n_atoms]
                )
                # Padding回原始大小
                pred_ts_padded = torch.zeros_like(r_pos[i])
                pred_ts_padded[:n_atoms] = pred_ts_single
                pred_ts_list.append(pred_ts_padded)
            
            pred_ts = torch.stack(pred_ts_list, dim=0)
            
            # 计算损失
            loss = criterion(pred_ts[mask], ts_pos[mask])
            
            # 计算RMSD和成功率
            batch_rmsd = 0
            for i in range(len(batch['num_atoms'])):
                n_atoms = batch['num_atoms'][i]
                pred = pred_ts[i, :n_atoms].cpu().numpy()
                true = ts_pos[i, :n_atoms].cpu().numpy()
                rmsd = calculate_rmsd(pred, true)
                batch_rmsd += rmsd
                total_count += 1
                if rmsd < threshold:
                    success_count += 1
            
            batch_rmsd /= len(batch['num_atoms'])
            
            total_loss += loss.item()
            total_rmsd += batch_rmsd
            count += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'rmsd': f'{batch_rmsd:.4f}'})
    
    success_rate = success_count / total_count * 100
    return total_loss / count, total_rmsd / count, success_rate


def main():
    parser = argparse.ArgumentParser(description='训练过渡态预测模型（XYZ格式）')
    parser.add_argument('--train_dir', type=str, default='./train_data',
                        help='训练数据目录')
    parser.add_argument('--output_dir', type=str, default='./outputs_xyz',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--save_every', type=int, default=5,
                        help='每N个epoch保存一次')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置设备
    device = setup_device()
    
    # 加载数据
    print("\n加载数据...")
    dataset = XYZTransitionStateDataset(args.train_dir)
    
    # 划分训练集和验证集（90% / 10%）
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda')
    )
    
    # 创建模型
    print("\n创建模型...")
    model = create_model().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # PyTorch 2.0+ 编译优化（CPU也支持）
    try:
        if hasattr(torch, 'compile'):
            print("✓ 应用torch.compile优化...")
            model = torch.compile(model, mode='default')
            print("✓ 模型编译完成（首次运行会较慢，后续会加速）")
    except Exception as e:
        print(f"⚠️  编译优化失败: {e}，使用常规模式")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    print("\n开始训练...\n")
    best_val_rmsd = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_rmsd = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        # 验证
        val_loss, val_rmsd, success_rate = validate(
            model, val_loader, criterion, device
        )
        
        # 学习率调整
        scheduler.step()
        
        # 记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('RMSD/train', train_rmsd, epoch)
        writer.add_scalar('RMSD/val', val_rmsd, epoch)
        writer.add_scalar('SuccessRate/val', success_rate, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 打印结果
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, RMSD: {train_rmsd:.4f} Å")
        print(f"  Val   Loss: {val_loss:.4f}, RMSD: {val_rmsd:.4f} Å, Success Rate: {success_rate:.2f}%")
        
        # 保存最佳模型
        if val_rmsd < best_val_rmsd:
            best_val_rmsd = val_rmsd
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmsd': val_rmsd,
                'success_rate': success_rate,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✓ 保存最佳模型 (Val RMSD: {val_rmsd:.4f} Å)")
        
        # 定期保存
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmsd': val_rmsd,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    writer.close()
    print("\n训练完成！")
    print(f"最佳验证RMSD: {best_val_rmsd:.4f} Å")


if __name__ == '__main__':
    main()

