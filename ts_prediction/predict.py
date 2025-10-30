"""
预测脚本 - 对测试集生成过渡态结构预测
"""
import os
import sys
import argparse
import torch
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ts_prediction.models import create_model
from ts_prediction.data import create_dataloader
from ts_prediction.utils.xyz_io import arrays_to_atoms, write_xyz


def predict(model, dataloader, device, output_dir):
    """
    对测试集进行预测
    
    Args:
        model: 训练好的模型
        dataloader: 测试数据加载器
        device: 设备
        output_dir: 输出目录
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始预测，结果将保存到: {output_dir}")
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Predict]')
        for batch in pbar:
            for sample in batch:
                reaction_id = sample['reaction_id']
                atomic_nums = sample['reactant_nums'].to(device)
                r_pos = sample['reactant_pos'].to(device)
                p_pos = sample['product_pos'].to(device)
                r_dist = sample['reactant_dist'].to(device)
                p_dist = sample['product_dist'].to(device)
                
                # 预测过渡态坐标
                ts_pos_pred, alpha = model(atomic_nums, r_pos, p_pos, r_dist, p_dist)
                
                # 转换为numpy
                atomic_nums_np = atomic_nums.cpu().numpy()
                ts_pos_np = ts_pos_pred.cpu().numpy()
                
                # 创建ASE Atoms对象
                ts_atoms = arrays_to_atoms(atomic_nums_np, ts_pos_np)
                
                # 保存为XYZ文件
                output_path = os.path.join(output_dir, f"{reaction_id}_ts_pred.xyz")
                write_xyz(output_path, ts_atoms)
                
                pbar.set_postfix({'reaction': reaction_id})
    
    print(f"\n预测完成！共生成 {len(dataloader.dataset)} 个预测文件")


def main(args):
    # 设置设备（优雅处理GPU不兼容）
    if torch.cuda.is_available():
        try:
            # 更严格的GPU兼容性测试
            import torch.nn as nn
            test_embed = nn.Embedding(10, 10).cuda()
            test_input = torch.randint(0, 10, (5,)).cuda()
            _ = test_embed(test_input)
            device = torch.device('cuda')
            print(f"✓ 使用设备: GPU - {torch.cuda.get_device_name(0)}")
            del test_embed, test_input
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"⚠ GPU检测到但不兼容: {str(e)[:80]}")
            print(f"⚠ 降级到CPU")
            device = torch.device('cpu')
            torch.set_num_threads(min(args.num_workers, 16))
    else:
        device = torch.device('cpu')
        torch.set_num_threads(min(args.num_workers, 16))
        print(f"使用设备: CPU（{torch.get_num_threads()}线程）")
    
    # 创建模型
    print("加载模型...")
    model = create_model()
    model = model.to(device)
    
    # 加载训练好的权重
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 加载模型权重: {args.checkpoint}")
        if 'train_rmsd' in checkpoint:
            print(f"  训练RMSD: {checkpoint['train_rmsd']:.4f} Å")
    else:
        print(f"错误: 找不到模型文件 {args.checkpoint}")
        return
    
    # 创建测试数据加载器
    print("加载测试数据...")
    test_loader = create_dataloader(
        data_root=args.test_data,
        batch_size=args.batch_size,
        shuffle=False,
        mode='test',
        num_workers=args.num_workers
    )
    
    # 预测
    predict(model, test_loader, device, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='过渡态结构预测')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--test_data', type=str, required=True,
                        help='测试数据根目录（包含rxn文件夹）')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='预测结果输出目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    args = parser.parse_args()
    
    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    main(args)

