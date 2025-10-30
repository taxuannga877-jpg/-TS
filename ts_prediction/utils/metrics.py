"""
评估指标 - RMSD计算（Kabsch对齐）
"""
import numpy as np
from scipy.spatial.transform import Rotation


def kabsch_align(P, Q):
    """
    使用Kabsch算法对齐两个点集
    将P对齐到Q
    
    Args:
        P: 待对齐坐标 shape (N, 3)
        Q: 参考坐标 shape (N, 3)
        
    Returns:
        P_aligned: 对齐后的坐标 shape (N, 3)
    """
    # 中心化
    P_center = P - P.mean(axis=0)
    Q_center = Q - Q.mean(axis=0)
    
    # 计算协方差矩阵
    H = P_center.T @ Q_center
    
    # SVD分解
    U, S, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = Vt.T @ U.T
    
    # 修正镜像（确保是旋转而不是反射）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 应用旋转和平移
    P_aligned = (R @ P_center.T).T + Q.mean(axis=0)
    
    return P_aligned


def calculate_rmsd(pred_pos, true_pos, align=True):
    """
    计算RMSD（均方根偏差）
    
    Args:
        pred_pos: 预测坐标 shape (N, 3)
        true_pos: 真实坐标 shape (N, 3)
        align: 是否进行Kabsch对齐
        
    Returns:
        rmsd: RMSD值（埃）
    """
    if align:
        pred_pos = kabsch_align(pred_pos, true_pos)
    
    diff = pred_pos - true_pos
    rmsd = np.sqrt((diff ** 2).sum() / len(pred_pos))
    
    return rmsd


def batch_rmsd(pred_positions, true_positions):
    """
    批量计算RMSD
    
    Args:
        pred_positions: 预测坐标列表 List[(N_i, 3)]
        true_positions: 真实坐标列表 List[(N_i, 3)]
        
    Returns:
        rmsds: RMSD数组
        mean_rmsd: 平均RMSD
        success_rate: 成功率 (RMSD <= 0.5Å)
    """
    rmsds = []
    for pred, true in zip(pred_positions, true_positions):
        rmsd = calculate_rmsd(pred, true, align=True)
        rmsds.append(rmsd)
    
    rmsds = np.array(rmsds)
    mean_rmsd = rmsds.mean()
    success_rate = (rmsds <= 0.5).mean()
    
    return rmsds, mean_rmsd, success_rate


