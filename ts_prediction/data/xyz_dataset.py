"""
XYZ格式的过渡态数据集
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ase.io import read as ase_read


class XYZTransitionStateDataset(Dataset):
    """从XYZ文件加载过渡态数据的数据集"""
    
    def __init__(self, data_dir, max_atoms=None):
        """
        Args:
            data_dir: 数据目录，包含rxnXXXX子文件夹
            max_atoms: 最大原子数，用于padding
        """
        self.data_dir = data_dir
        self.reactions = []
        
        # 扫描所有反应文件夹
        for rxn_folder in sorted(os.listdir(data_dir)):
            rxn_path = os.path.join(data_dir, rxn_folder)
            if not os.path.isdir(rxn_path):
                continue
            
            r_path = os.path.join(rxn_path, 'r.xyz')
            p_path = os.path.join(rxn_path, 'p.xyz')
            ts_path = os.path.join(rxn_path, 'ts.xyz')
            
            if os.path.exists(r_path) and os.path.exists(p_path) and os.path.exists(ts_path):
                self.reactions.append({
                    'id': rxn_folder,
                    'r_path': r_path,
                    'p_path': p_path,
                    'ts_path': ts_path
                })
        
        print(f"找到 {len(self.reactions)} 个完整反应")
        
        # 确定最大原子数
        if max_atoms is None:
            self.max_atoms = self._find_max_atoms()
        else:
            self.max_atoms = max_atoms
        
        print(f"最大原子数: {self.max_atoms}")
    
    def _find_max_atoms(self):
        """找到数据集中的最大原子数"""
        max_atoms = 0
        for rxn in self.reactions[:100]:  # 只检查前100个以加快速度
            atoms_r = ase_read(rxn['r_path'])
            max_atoms = max(max_atoms, len(atoms_r))
        return max_atoms
    
    def __len__(self):
        return len(self.reactions)
    
    def __getitem__(self, idx):
        rxn = self.reactions[idx]
        
        # 读取XYZ文件
        atoms_r = ase_read(rxn['r_path'])
        atoms_p = ase_read(rxn['p_path'])
        atoms_ts = ase_read(rxn['ts_path'])
        
        # 提取数据
        atomic_numbers_r = atoms_r.get_atomic_numbers()
        positions_r = atoms_r.get_positions()
        
        atomic_numbers_p = atoms_p.get_atomic_numbers()
        positions_p = atoms_p.get_positions()
        
        atomic_numbers_ts = atoms_ts.get_atomic_numbers()
        positions_ts = atoms_ts.get_positions()
        
        num_atoms = len(atomic_numbers_r)
        
        # 验证原子数一致
        assert len(atomic_numbers_p) == num_atoms, f"反应物和产物原子数不一致: {rxn['id']}"
        assert len(atomic_numbers_ts) == num_atoms, f"过渡态原子数不一致: {rxn['id']}"
        
        # 转换为tensor
        atomic_numbers = torch.from_numpy(atomic_numbers_r).long()
        r_pos = torch.from_numpy(positions_r).float()
        p_pos = torch.from_numpy(positions_p).float()
        ts_pos = torch.from_numpy(positions_ts).float()
        
        return {
            'rxn_id': rxn['id'],
            'atomic_numbers': atomic_numbers,
            'num_atoms': num_atoms,
            'r_pos': r_pos,
            'p_pos': p_pos,
            'ts_pos': ts_pos
        }


def collate_fn(batch):
    """自定义collate函数，处理变长分子"""
    max_atoms = max([item['num_atoms'] for item in batch])
    
    batch_size = len(batch)
    
    # 初始化padded tensors
    atomic_numbers = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    r_pos = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    p_pos = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    ts_pos = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
    num_atoms_list = []
    rxn_ids = []
    
    for i, item in enumerate(batch):
        n = item['num_atoms']
        atomic_numbers[i, :n] = item['atomic_numbers']
        r_pos[i, :n] = item['r_pos']
        p_pos[i, :n] = item['p_pos']
        ts_pos[i, :n] = item['ts_pos']
        mask[i, :n] = True
        num_atoms_list.append(n)
        rxn_ids.append(item['rxn_id'])
    
    return {
        'rxn_ids': rxn_ids,
        'atomic_numbers': atomic_numbers,
        'r_pos': r_pos,
        'p_pos': p_pos,
        'ts_pos': ts_pos,
        'mask': mask,
        'num_atoms': num_atoms_list
    }


def create_xyz_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=4):
    """创建数据加载器"""
    dataset = XYZTransitionStateDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return dataloader


