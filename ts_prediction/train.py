"""
训练脚本 - 过渡态预测模型
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ts_prediction.models import create_model
from ts_prediction.data import create_dataloader
from ts_prediction.utils.metrics import calculate_rmsd


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    rmsd_list = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        batch_loss = 0
        batch_rmsd = []
        
        # 逐个处理batch中的样本（因为原子数不同）
        for sample in batch:
            atomic_nums = sample['reactant_nums'].to(device)
            r_pos = sample['reactant_pos'].to(device)
            p_pos = sample['product_pos'].to(device)
            ts_pos_true = sample['ts_pos'].to(device)
            r_dist = sample['reactant_dist'].to(device)
            p_dist = sample['product_dist'].to(device)
            
            # 前向传播
            ts_pos_pred, alpha = model(atomic_nums, r_pos, p_pos, r_dist, p_dist)
            
            # 计算损失（MSE）
            loss = criterion(ts_pos_pred, ts_pos_true)
            batch_loss += loss
            
            # 计算RMSD（用于监控）
            with torch.no_grad():
                rmsd = calculate_rmsd(
                    ts_pos_pred.cpu().numpy(),
                    ts_pos_true.cpu().numpy(),
                    align=True
                )
                batch_rmsd.append(rmsd)
        
        # 平均损失
        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += batch_loss.item()
        rmsd_list.extend(batch_rmsd)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{batch_loss.item():.4f}',
            'rmsd': f'{np.mean(batch_rmsd):.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_rmsd = np.mean(rmsd_list)
    
    return avg_loss, avg_rmsd


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    rmsd_list = []
    success_count = 0
    total_count = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Validate]')
        for batch in pbar:
            batch_loss = 0
            
            for sample in batch:
                atomic_nums = sample['reactant_nums'].to(device)
                r_pos = sample['reactant_pos'].to(device)
                p_pos = sample['product_pos'].to(device)
                ts_pos_true = sample['ts_pos'].to(device)
                r_dist = sample['reactant_dist'].to(device)
                p_dist = sample['product_dist'].to(device)
                
                # 前向传播
                ts_pos_pred, alpha = model(atomic_nums, r_pos, p_pos, r_dist, p_dist)
                
                # 计算损失
                loss = criterion(ts_pos_pred, ts_pos_true)
                batch_loss += loss
                
                # 计算RMSD
                rmsd = calculate_rmsd(
                    ts_pos_pred.cpu().numpy(),
                    ts_pos_true.cpu().numpy(),
                    align=True
                )
                rmsd_list.append(rmsd)
                
                # 统计成功率（RMSD <= 0.5Å）
                if rmsd <= 0.5:
                    success_count += 1
                total_count += 1
            
            batch_loss = batch_loss / len(batch)
            total_loss += batch_loss.item()
            
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'rmsd': f'{np.mean(rmsd_list):.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_rmsd = np.mean(rmsd_list)
    success_rate = success_count / total_count if total_count > 0 else 0
    
    return avg_loss, avg_rmsd, success_rate


def main(args):
    # 设置设备（优雅处理GPU不兼容）
    if torch.cuda.is_available():
        try:
            # 更严格的GPU兼容性测试
            test_embed = nn.Embedding(10, 10).cuda()
            test_input = torch.randint(0, 10, (5,)).cuda()
            _ = test_embed(test_input)
            device = torch.device('cuda')
            print(f"✓ 使用设备: GPU - {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            del test_embed, test_input
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"⚠ GPU检测到但不兼容: {str(e)[:80]}")
            print(f"⚠ 降级到CPU（充分利用多核）")
            device = torch.device('cpu')
            # CPU性能优化
            torch.set_num_threads(min(args.num_workers, 16))
            print(f"  CPU线程数: {torch.get_num_threads()}")
    else:
        device = torch.device('cpu')
        torch.set_num_threads(min(args.num_workers, 16))
        print(f"使用设备: CPU（{torch.get_num_threads()}线程）")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # 创建数据加载器
    print("加载训练数据...")
    train_loader = create_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        shuffle=True,
        mode='train',
        num_workers=args.num_workers
    )
    
    # 创建模型
    print("创建模型...")
    model = create_model()
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # 训练循环
    best_rmsd = float('inf')
    print(f"\n开始训练 {args.epochs} 个epochs...")
    print(f"批大小: {args.batch_size}, 学习率: {args.lr}\n")
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_rmsd = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, RMSD: {train_rmsd:.4f} Å")
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('RMSD/train', train_rmsd, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 更新学习率
        scheduler.step(train_loss)
        
        # 保存最佳模型
        if train_rmsd < best_rmsd:
            best_rmsd = train_rmsd
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_rmsd': train_rmsd,
            }, checkpoint_path)
            print(f"  ✓ 保存最佳模型 (RMSD: {train_rmsd:.4f} Å)")
        
        # 定期保存checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_rmsd': train_rmsd,
            }, checkpoint_path)
            print(f"  ✓ 保存checkpoint: epoch_{epoch}.pth")
    
    writer.close()
    print(f"\n训练完成！最佳RMSD: {best_rmsd:.4f} Å")
    print(f"模型保存在: {os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练过渡态预测模型')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                        help='训练数据根目录（包含rxn0000等文件夹）')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--save_every', type=int, default=10,
                        help='每N个epoch保存一次')
    
    # GPU
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    args = parser.parse_args()
    
    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    main(args)

