"""
TS预测的评估指标
"""
import numpy as np
import torch


def calculate_rmsd(pred_coords, true_coords):
    """
    计算预测坐标和真实坐标之间的均方根偏差（RMSD）
    
    参数：
        pred_coords (np.ndarray or torch.Tensor): 预测坐标，形状 (N, 3)
        true_coords (np.ndarray or torch.Tensor): 真实坐标，形状 (N, 3)
    
    返回：
        float: RMSD值，单位为埃（Angstroms）
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.cpu().numpy()
    if isinstance(true_coords, torch.Tensor):
        true_coords = true_coords.cpu().numpy()
    
    diff = pred_coords - true_coords
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=-1)))
    return rmsd


def calculate_mae(pred_coords, true_coords):
    """
    计算预测坐标和真实坐标之间的平均绝对误差（MAE）
    
    参数：
        pred_coords (np.ndarray or torch.Tensor): 预测坐标，形状 (N, 3)
        true_coords (np.ndarray or torch.Tensor): 真实坐标，形状 (N, 3)
    
    返回：
        float: MAE值，单位为埃（Angstroms）
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.cpu().numpy()
    if isinstance(true_coords, torch.Tensor):
        true_coords = true_coords.cpu().numpy()
    
    diff = np.abs(pred_coords - true_coords)
    mae = np.mean(diff)
    return mae


def calculate_success_rate(rmsds, threshold=0.5):
    """
    计算成功率（RMSD < 阈值的预测百分比）
    
    参数：
        rmsds (list or np.ndarray): RMSD值列表
        threshold (float): 成功阈值，单位为埃（默认：0.5）
    
    返回：
        float: 成功率，百分比形式（0-100）
    """
    rmsds = np.array(rmsds)
    success_count = np.sum(rmsds < threshold)
    success_rate = (success_count / len(rmsds)) * 100
    return success_rate


def kabsch_alignment(pred, target):
    """
    使用Kabsch算法将预测坐标对齐到目标坐标
    
    参数：
        pred (np.ndarray or torch.Tensor): 预测坐标，形状 (N, 3)
        target (np.ndarray or torch.Tensor): 目标坐标，形状 (N, 3)
    
    返回：
        np.ndarray or torch.Tensor: 对齐后的预测坐标
    """
    is_torch = isinstance(pred, torch.Tensor)
    
    if is_torch:
        device = pred.device
        dtype = pred.dtype
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
    else:
        pred_np = pred
        target_np = target
    
    # 中心化坐标
    pred_centered = pred_np - pred_np.mean(axis=0)
    target_centered = target_np - target_np.mean(axis=0)
    
    # 使用SVD计算旋转矩阵
    H = pred_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 修正反射
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 应用旋转和平移
    aligned = pred_centered @ R + target_np.mean(axis=0)
    
    if is_torch:
        aligned = torch.from_numpy(aligned).to(device=device, dtype=dtype)
    
    return aligned

