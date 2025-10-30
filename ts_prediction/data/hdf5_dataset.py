"""
HDF5格式的过渡态预测数据集
用于读取Transition1x.h5文件
"""
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class HDF5TransitionStateDataset(Dataset):
    """
    从HDF5文件加载过渡态数据
    
    数据结构:
    h5文件['train'|'val'|'test'] / {分子式} / {rxn_id} /
        - atomic_numbers
        - reactant/positions
        - product/positions
        - transition_state/positions (仅训练集和验证集)
    """
    
    def __init__(self, hdf5_path, split='train'):
        """
        Args:
            hdf5_path: HDF5文件路径
            split: 'train', 'val', 或 'test'
        """
        self.hdf5_path = hdf5_path
        self.split = split
        
        # 收集所有反应的索引
        self.reaction_indices = []
        
        with h5py.File(hdf5_path, 'r') as f:
            split_group = f[split]
            
            for formula in split_group.keys():
                formula_group = split_group[formula]
                for rxn_id in formula_group.keys():
                    self.reaction_indices.append((formula, rxn_id))
        
        print(f"[{split}] 加载了 {len(self.reaction_indices)} 个反应")
    
    def __len__(self):
        return len(self.reaction_indices)
    
    def __getitem__(self, idx):
        """
        返回单个反应的数据
        
        Returns:
            data_dict: {
                'reactant_nums': 原子序数,
                'reactant_pos': 反应物坐标,
                'product_pos': 产物坐标,
                'ts_pos': 过渡态坐标 (仅训练集和验证集),
                'reactant_dist': 反应物距离矩阵,
                'product_dist': 产物距离矩阵,
                'reaction_id': 反应ID,
                'num_atoms': 原子数
            }
        """
        formula, rxn_id = self.reaction_indices[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            rxn_group = f[self.split][formula][rxn_id]
            
            # 读取原子序数
            atomic_nums = rxn_group['atomic_numbers'][:]
            
            # 读取坐标（移除batch维度 (1, N, 3) -> (N, 3)）
            r_pos = rxn_group['reactant']['positions'][0]  # (N, 3)
            p_pos = rxn_group['product']['positions'][0]   # (N, 3)
            
            # 计算距离矩阵
            r_dist = self._get_distance_matrix(r_pos)
            p_dist = self._get_distance_matrix(p_pos)
            
            data_dict = {
                'reactant_nums': torch.from_numpy(atomic_nums).long(),
                'reactant_pos': torch.from_numpy(r_pos).float(),
                'product_pos': torch.from_numpy(p_pos).float(),
                'reactant_dist': torch.from_numpy(r_dist).float(),
                'product_dist': torch.from_numpy(p_dist).float(),
                'reaction_id': f"{formula}_{rxn_id}",
                'num_atoms': len(atomic_nums)
            }
            
            # 训练集和验证集有过渡态标签
            if self.split in ['train', 'val']:
                ts_pos = rxn_group['transition_state']['positions'][0]
                data_dict['ts_pos'] = torch.from_numpy(ts_pos).float()
        
        return data_dict
    
    @staticmethod
    def _get_distance_matrix(positions):
        """
        计算距离矩阵
        
        Args:
            positions: (N, 3) numpy array
            
        Returns:
            dist_matrix: (N, N) numpy array
        """
        diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
        dist_matrix = np.linalg.norm(diff, axis=-1)  # (N, N)
        return dist_matrix


def collate_fn(batch):
    """
    自定义batch整理函数
    由于不同反应的原子数不同，保持list形式
    """
    return batch


def create_hdf5_dataloader(hdf5_path, split='train', batch_size=32, 
                           shuffle=True, num_workers=4):
    """
    创建HDF5数据加载器
    
    Args:
        hdf5_path: HDF5文件路径
        split: 'train', 'val', 或 'test'
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        
    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = HDF5TransitionStateDataset(hdf5_path, split=split)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


