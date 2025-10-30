"""
预测测试集的过渡态结构
"""
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from ase.io import read as ase_read, write as ase_write
from ase import Atoms

from ts_prediction.models import create_model


def setup_device():
    """设置计算设备"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        torch.set_num_threads(os.cpu_count())
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
        torch.set_num_threads(os.cpu_count())
        return torch.device('cpu')


def calc_dist_matrix_single(pos):
    """计算单个分子的距离矩阵"""
    # pos: (N, 3)
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 3)
    dist = torch.norm(diff, dim=-1)  # (N, N)
    return dist


def predict_single_reaction(model, atoms_r, atoms_p, device):
    """预测单个反应的过渡态"""
    model.eval()
    
    with torch.no_grad():
        # 提取数据
        atomic_numbers = torch.from_numpy(atoms_r.get_atomic_numbers()).long().to(device)
        r_pos = torch.from_numpy(atoms_r.get_positions()).float().to(device)
        p_pos = torch.from_numpy(atoms_p.get_positions()).float().to(device)
        
        # 计算距离矩阵
        r_dist = calc_dist_matrix_single(r_pos)
        p_dist = calc_dist_matrix_single(p_pos)
        
        # 预测
        pred_ts, _ = model(atomic_numbers, r_pos, p_pos, r_dist, p_dist)
        
        # 转换为numpy
        pred_positions = pred_ts.cpu().numpy()
    
    return pred_positions


def main():
    parser = argparse.ArgumentParser(description='预测测试集的过渡态结构')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='测试数据目录（包含rxnX子文件夹）')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录（将创建rxnX子文件夹）')
    parser.add_argument('--reactant_file', type=str, default='RS.xyz',
                        help='反应物文件名（默认: RS.xyz）')
    parser.add_argument('--product_file', type=str, default='PS.xyz',
                        help='产物文件名（默认: PS.xyz）')
    parser.add_argument('--output_file', type=str, default='TS_pred.xyz',
                        help='输出过渡态文件名（默认: TS_pred.xyz）')
    args = parser.parse_args()
    
    # 设置设备
    device = setup_device()
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    model = create_model().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功")
    if 'val_rmsd' in checkpoint:
        print(f"  验证集RMSD: {checkpoint['val_rmsd']:.4f} Å")
    if 'success_rate' in checkpoint:
        print(f"  验证集成功率: {checkpoint['success_rate']:.2f}%")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 扫描测试数据
    test_folders = sorted([f for f in os.listdir(args.test_dir) 
                          if os.path.isdir(os.path.join(args.test_dir, f))])
    
    print(f"\n找到 {len(test_folders)} 个测试反应")
    print(f"开始预测...\n")
    
    success_count = 0
    failed_reactions = []
    
    for rxn_folder in tqdm(test_folders, desc='预测进度'):
        try:
            # 读取反应物和产物
            rxn_path = os.path.join(args.test_dir, rxn_folder)
            r_path = os.path.join(rxn_path, args.reactant_file)
            p_path = os.path.join(rxn_path, args.product_file)
            
            if not os.path.exists(r_path):
                print(f"⚠️  {rxn_folder}: 找不到 {args.reactant_file}")
                failed_reactions.append(rxn_folder)
                continue
            
            if not os.path.exists(p_path):
                print(f"⚠️  {rxn_folder}: 找不到 {args.product_file}")
                failed_reactions.append(rxn_folder)
                continue
            
            atoms_r = ase_read(r_path)
            atoms_p = ase_read(p_path)
            
            # 验证原子数一致
            if len(atoms_r) != len(atoms_p):
                print(f"⚠️  {rxn_folder}: 反应物和产物原子数不一致")
                failed_reactions.append(rxn_folder)
                continue
            
            # 预测过渡态
            pred_positions = predict_single_reaction(model, atoms_r, atoms_p, device)
            
            # 创建过渡态Atoms对象
            pred_atoms = Atoms(
                symbols=atoms_r.get_chemical_symbols(),
                positions=pred_positions
            )
            
            # 保存预测结果到test_data_1的rxn文件夹中
            output_path = os.path.join(rxn_path, args.output_file)
            
            ase_write(output_path, pred_atoms, format='xyz')
            success_count += 1
            
        except Exception as e:
            print(f"❌ {rxn_folder}: 预测失败 - {e}")
            failed_reactions.append(rxn_folder)
    
    # 打印总结
    print(f"\n" + "="*60)
    print(f"预测完成！")
    print(f"="*60)
    print(f"总反应数: {len(test_folders)}")
    print(f"成功预测: {success_count}")
    print(f"失败: {len(failed_reactions)}")
    print(f"成功率: {success_count/len(test_folders)*100:.2f}%")
    
    if failed_reactions:
        print(f"\n失败的反应:")
        for rxn in failed_reactions[:10]:
            print(f"  - {rxn}")
        if len(failed_reactions) > 10:
            print(f"  ... 还有 {len(failed_reactions)-10} 个")
    
    print(f"\n输出目录: {args.output_dir}")
    print(f"准备提交: 将 {args.output_dir} 压缩为zip文件")


if __name__ == '__main__':
    main()

