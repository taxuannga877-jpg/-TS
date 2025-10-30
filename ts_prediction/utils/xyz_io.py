"""
XYZ文件读写工具 - 使用ASE库
"""
import numpy as np
from ase.io import read, write
from ase import Atoms


def read_xyz(filepath):
    """
    读取XYZ文件
    
    Args:
        filepath: XYZ文件路径
        
    Returns:
        atoms: ASE Atoms对象
    """
    return read(filepath)


def write_xyz(filepath, atoms):
    """
    写入XYZ文件
    
    Args:
        filepath: 输出文件路径
        atoms: ASE Atoms对象
    """
    write(filepath, atoms)


def atoms_to_arrays(atoms):
    """
    将ASE Atoms转换为numpy数组
    
    Args:
        atoms: ASE Atoms对象
        
    Returns:
        atomic_numbers: 原子序数 shape (N,)
        positions: 原子坐标 shape (N, 3)
    """
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    return atomic_numbers, positions


def arrays_to_atoms(atomic_numbers, positions):
    """
    从numpy数组创建ASE Atoms对象
    
    Args:
        atomic_numbers: 原子序数 shape (N,)
        positions: 原子坐标 shape (N, 3)
        
    Returns:
        atoms: ASE Atoms对象
    """
    return Atoms(numbers=atomic_numbers, positions=positions)


def get_distance_matrix(positions):
    """
    计算原子间距离矩阵
    
    Args:
        positions: 原子坐标 shape (N, 3)
        
    Returns:
        dist_matrix: 距离矩阵 shape (N, N)
    """
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    dist_matrix = np.linalg.norm(diff, axis=-1)  # (N, N)
    return dist_matrix


