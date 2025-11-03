"""
TSFF混合方案 - 数据加载器
结合RDKit化学特征 + PyG图结构 + 3D几何信息
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.io import read as ase_read

# 禁用RDKit警告日志
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# 原子序数映射
ATOM_TO_Z = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53
}


def read_xyz_file(xyz_path):
    """读取XYZ文件，返回原子符号和坐标"""
    atoms = ase_read(xyz_path)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    return symbols, positions


def xyz_to_rdkit_mol(xyz_path, try_sanitize=True):
    """
    从XYZ文件创建RDKit分子对象
    自动推断化学键
    """
    symbols, positions = read_xyz_file(xyz_path)
    
    # 创建RDKit分子
    mol = Chem.RWMol()
    for symbol in symbols:
        atom = Chem.Atom(symbol)
        mol.AddAtom(atom)
    
    # 添加3D坐标
    conf = Chem.Conformer(len(symbols))
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, tuple(pos))
    mol.AddConformer(conf)
    
    # 推断化学键（基于距离）
    try:
        # 尝试使用新版API
        if hasattr(Chem, 'rdDetermineBonds'):
            Chem.rdDetermineBonds.DetermineConnectivity(mol)
        elif hasattr(Chem.rdDetermineBonds, 'DetermineConnectivity'):
            from rdkit.Chem import rdDetermineBonds
            rdDetermineBonds.DetermineConnectivity(mol)
        else:
            raise AttributeError("DetermineConnectivity not available")
        
        if try_sanitize:
            try:
                Chem.SanitizeMol(mol)
            except:
                pass
    except Exception as e:
        # 使用简单距离判断
        mol = simple_bond_inference(symbols, positions)
    
    return mol


def simple_bond_inference(symbols, positions, bond_length_tolerance=1.3):
    """
    简单的键推断：基于原子间距离
    """
    # 典型键长（Å）
    TYPICAL_BOND_LENGTHS = {
        ('C', 'C'): 1.54, ('C', 'N'): 1.47, ('C', 'O'): 1.43,
        ('C', 'H'): 1.09, ('N', 'H'): 1.01, ('O', 'H'): 0.96,
        ('C', 'S'): 1.82, ('C', 'P'): 1.85, ('N', 'N'): 1.45,
        ('O', 'O'): 1.48, ('C', 'F'): 1.35, ('C', 'Cl'): 1.77,
    }
    
    mol = Chem.RWMol()
    for symbol in symbols:
        mol.AddAtom(Chem.Atom(symbol))
    
    # 添加坐标
    conf = Chem.Conformer(len(symbols))
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, tuple(pos))
    mol.AddConformer(conf)
    
    # 添加键
    n_atoms = len(symbols)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            pair = tuple(sorted([symbols[i], symbols[j]]))
            
            # 获取参考键长
            if pair in TYPICAL_BOND_LENGTHS:
                ref_length = TYPICAL_BOND_LENGTHS[pair]
            else:
                ref_length = 1.5  # 默认值
            
            # 判断是否成键
            if dist < ref_length * bond_length_tolerance:
                mol.AddBond(i, j, Chem.BondType.SINGLE)
    
    return mol.GetMol()


def extract_rdkit_features(mol):
    """
    提取RDKit化学特征
    返回: (N_atoms, feature_dim) 的特征矩阵
    """
    # 确保分子已经过sanitize
    try:
        Chem.SanitizeMol(mol)
    except:
        pass
    
    features = []
    
    for atom in mol.GetAtoms():
        # 原子特征（使用安全的方法）
        try:
            degree = atom.GetDegree()
        except:
            degree = 0
        
        try:
            num_hs = atom.GetTotalNumHs()
        except:
            num_hs = 0
            
        try:
            hybridization = int(atom.GetHybridization())
        except:
            hybridization = 0
        
        feat = [
            atom.GetAtomicNum(),                    # 原子序数
            degree,                                 # 度数
            atom.GetFormalCharge(),                 # 形式电荷
            num_hs,                                 # 氢原子数
            int(atom.GetIsAromatic()),              # 是否芳香
            hybridization,                          # 杂化轨道
            int(atom.IsInRing()),                   # 是否在环中
            atom.GetMass(),                         # 原子质量
        ]
        features.append(feat)
    
    return torch.tensor(features, dtype=torch.float32)


def create_pyg_graph(xyz_path, cutoff=5.0, max_neighbors=32):
    """
    从XYZ文件创建PyG图对象
    
    返回:
        data: PyG Data对象
        mol: RDKit分子对象
    """
    # 读取XYZ
    symbols, positions = read_xyz_file(xyz_path)
    
    # 转换为torch tensor
    z = torch.tensor([ATOM_TO_Z.get(s, 0) for s in symbols], dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float32)
    
    # 构建边（基于距离cutoff）
    edge_index = radius_graph(pos, r=cutoff, max_num_neighbors=max_neighbors)
    
    # 提取RDKit化学特征
    try:
        mol = xyz_to_rdkit_mol(xyz_path)
        chem_feat = extract_rdkit_features(mol)
    except Exception as e:
        print(f"⚠️ RDKit特征提取失败: {e}，使用零特征")
        chem_feat = torch.zeros(len(symbols), 8)
        mol = None
    
    # 创建PyG Data对象
    data = Data(
        z=z,                    # 原子序数 (N,)
        pos=pos,                # 3D坐标 (N, 3)
        chem_feat=chem_feat,    # 化学特征 (N, 8)
        edge_index=edge_index,  # 边索引 (2, E)
    )
    
    return data, mol


class TSFFHybridDataset(Dataset):
    """
    TSFF混合方案数据集
    加载R, P, TS的XYZ文件，转换为PyG图对象
    """
    
    def __init__(self, data_dir, cutoff=5.0, augment=False):
        """
        Args:
            data_dir: 数据目录（包含rxn0000, rxn0001, ...子目录）
            cutoff: 距离cutoff（Å）
            augment: 是否数据增强
        """
        self.data_dir = data_dir
        self.cutoff = cutoff
        self.augment = augment
        
        # 扫描所有反应
        self.reactions = []
        for rxn_name in sorted(os.listdir(data_dir)):
            rxn_path = os.path.join(data_dir, rxn_name)
            if not os.path.isdir(rxn_path):
                continue
            
            r_path = os.path.join(rxn_path, 'r.xyz')
            p_path = os.path.join(rxn_path, 'p.xyz')
            ts_path = os.path.join(rxn_path, 'ts.xyz')
            
            if os.path.exists(r_path) and os.path.exists(p_path) and os.path.exists(ts_path):
                self.reactions.append({
                    'name': rxn_name,
                    'r': r_path,
                    'p': p_path,
                    'ts': ts_path
                })
        
        print(f"📊 加载了 {len(self.reactions)} 个反应")
    
    def __len__(self):
        return len(self.reactions)
    
    def __getitem__(self, idx):
        rxn = self.reactions[idx]
        
        # 创建PyG图
        data_r, mol_r = create_pyg_graph(rxn['r'], cutoff=self.cutoff)
        data_p, mol_p = create_pyg_graph(rxn['p'], cutoff=self.cutoff)
        data_ts, mol_ts = create_pyg_graph(rxn['ts'], cutoff=self.cutoff)
        
        # 数据增强（随机旋转）
        if self.augment:
            rotation = random_rotation_matrix()
            data_r.pos = data_r.pos @ rotation
            data_p.pos = data_p.pos @ rotation
            data_ts.pos = data_ts.pos @ rotation
        
        return {
            'name': rxn['name'],
            'data_r': data_r,
            'data_p': data_p,
            'data_ts': data_ts,
            'num_atoms': len(data_r.z)
        }


def random_rotation_matrix():
    """生成随机旋转矩阵"""
    angles = np.random.uniform(0, 2*np.pi, size=3)
    
    # X轴旋转
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])
    
    # Y轴旋转
    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    
    # Z轴旋转
    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx
    return torch.tensor(R, dtype=torch.float32)


def collate_fn(batch):
    """
    自定义collate函数，处理不同大小的分子
    """
    from torch_geometric.data import Batch as PyGBatch
    
    names = [item['name'] for item in batch]
    num_atoms = [item['num_atoms'] for item in batch]
    
    # 批处理PyG图
    data_r_batch = PyGBatch.from_data_list([item['data_r'] for item in batch])
    data_p_batch = PyGBatch.from_data_list([item['data_p'] for item in batch])
    data_ts_batch = PyGBatch.from_data_list([item['data_ts'] for item in batch])
    
    return {
        'names': names,
        'num_atoms': num_atoms,
        'data_r': data_r_batch,
        'data_p': data_p_batch,
        'data_ts': data_ts_batch
    }


# 测试代码
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './train_data'
    
    print(f"测试数据加载: {data_dir}")
    
    # 测试单个文件
    test_xyz = os.path.join(data_dir, 'rxn0000', 'r.xyz')
    if os.path.exists(test_xyz):
        print(f"\n测试文件: {test_xyz}")
        data, mol = create_pyg_graph(test_xyz)
        print(f"  原子数: {len(data.z)}")
        print(f"  边数: {data.edge_index.shape[1]}")
        print(f"  化学特征维度: {data.chem_feat.shape}")
        if mol:
            print(f"  RDKit分子: {Chem.MolToSmiles(mol)}")
    
    # 测试数据集
    print(f"\n测试数据集...")
    dataset = TSFFHybridDataset(data_dir, cutoff=5.0, augment=False)
    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n第一个样本:")
        print(f"  反应名称: {sample['name']}")
        print(f"  原子数: {sample['num_atoms']}")
        print(f"  R图: z={sample['data_r'].z.shape}, pos={sample['data_r'].pos.shape}")

