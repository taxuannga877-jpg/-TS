"""
过渡态预测数据集
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from ..utils.xyz_io import read_xyz, atoms_to_arrays, get_distance_matrix


class TransitionStateDataset(Dataset):
    """
    过渡态预测数据集
    每个样本包含: 反应物(r.xyz), 产物(p.xyz), 过渡态(ts.xyz)
    """
    
    def __init__(self, data_root, mode='train'):
        """
        Args:
            data_root: 数据根目录，包含rxn0000, rxn0001...文件夹
            mode: 'train' 或 'test'
        """
        self.data_root = data_root
        self.mode = mode
        
        # 查找所有反应文件夹
        self.reaction_dirs = []
        for dirname in sorted(os.listdir(data_root)):
            if dirname.startswith('rxn'):
                rxn_path = os.path.join(data_root, dirname)
                if os.path.isdir(rxn_path):
                    self.reaction_dirs.append(rxn_path)
        
        print(f"[{mode}] 找到 {len(self.reaction_dirs)} 个反应")
    
    def __len__(self):
        return len(self.reaction_dirs)
    
    def __getitem__(self, idx):
        """
        返回单个反应的数据
        
        Returns:
            data_dict: {
                'reactant_nums': 原子序数,
                'reactant_pos': 反应物坐标,
                'product_pos': 产物坐标,
                'ts_pos': 过渡态坐标 (仅训练集),
                'reactant_dist': 反应物距离矩阵,
                'product_dist': 产物距离矩阵,
                'reaction_id': 反应ID
            }
        """
        rxn_dir = self.reaction_dirs[idx]
        reaction_id = os.path.basename(rxn_dir)
        
        # 读取反应物
        r_atoms = read_xyz(os.path.join(rxn_dir, 'r.xyz'))
        r_nums, r_pos = atoms_to_arrays(r_atoms)
        
        # 读取产物
        p_atoms = read_xyz(os.path.join(rxn_dir, 'p.xyz'))
        p_nums, p_pos = atoms_to_arrays(p_atoms)
        
        # 计算距离矩阵（重要特征）
        r_dist = get_distance_matrix(r_pos)
        p_dist = get_distance_matrix(p_pos)
        
        data_dict = {
            'reactant_nums': torch.from_numpy(r_nums).long(),
            'reactant_pos': torch.from_numpy(r_pos).float(),
            'product_pos': torch.from_numpy(p_pos).float(),
            'reactant_dist': torch.from_numpy(r_dist).float(),
            'product_dist': torch.from_numpy(p_dist).float(),
            'reaction_id': reaction_id,
            'num_atoms': len(r_nums)
        }
        
        # 训练集读取过渡态
        if self.mode == 'train':
            ts_path = os.path.join(rxn_dir, 'ts.xyz')
            if os.path.exists(ts_path):
                ts_atoms = read_xyz(ts_path)
                ts_nums, ts_pos = atoms_to_arrays(ts_atoms)
                data_dict['ts_pos'] = torch.from_numpy(ts_pos).float()
        
        return data_dict


def collate_fn(batch):
    """
    自定义batch整理函数（处理不同原子数的分子）
    
    由于不同反应的原子数不同，我们需要padding或者保持list形式
    这里采用list形式，模型内部逐个处理
    """
    # 保持list形式，因为每个反应原子数可能不同
    return batch


def create_dataloader(data_root, batch_size=32, shuffle=True, mode='train', num_workers=4):
    """
    创建数据加载器
    
    Args:
        data_root: 数据根目录
        batch_size: 批大小
        shuffle: 是否打乱
        mode: 'train' 或 'test'
        num_workers: 数据加载线程数
        
    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = TransitionStateDataset(data_root, mode=mode)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


