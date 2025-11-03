"""
反应中心检测器模块

该模块实现了一个检测器，通过分析反应物和产物之间的坐标位移、
化学特征变化和局部环境变化，来识别参与化学反应的原子（即反应中心原子）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReactionCenterDetector(nn.Module):
    """
    反应中心检测器
    
    通过对每个原子的反应性进行评分来识别哪些原子参与化学反应，评分基于：
    1. R和P之间的坐标位移
    2. 化学特征变化
    3. 局部环境变化
    
    参数：
        chem_feat_dim (int): 化学特征维度（默认：8）
        hidden_dim (int): MLP隐藏层维度（默认：128）
    """
    
    def __init__(self, chem_feat_dim=8, hidden_dim=128):
        super().__init__()
        
        # 输入特征：位移(1) + 化学变化(chem_feat_dim) + 环境变化(64)
        input_dim = 1 + chem_feat_dim + 64
        
        # MLP用于预测反应性得分 [0, 1]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 局部环境编码器
        self.env_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
    
    def forward(self, pos_r, pos_p, chem_feat_r, chem_feat_p, batch):
        """
        前向传播，计算每个原子的反应得分
        
        参数：
            pos_r (Tensor): 反应物坐标，形状 (N, 3)
            pos_p (Tensor): 产物坐标，形状 (N, 3)
            chem_feat_r (Tensor): 反应物化学特征，形状 (N, chem_feat_dim)
            chem_feat_p (Tensor): 产物化学特征，形状 (N, chem_feat_dim)
            batch (Tensor): 批次索引，形状 (N,)
        
        返回：
            reaction_scores (Tensor): 每个原子的反应性得分，形状 (N,)，范围 [0, 1]
                                     更高的得分表示更可能是反应中心
        """
        # 1. 坐标位移大小
        displacement = torch.norm(pos_p - pos_r, dim=-1, keepdim=True)  # (N, 1)
        
        # 2. 化学特征变化
        chem_change = chem_feat_p - chem_feat_r  # (N, chem_feat_dim)
        
        # 3. 局部几何变化
        delta_pos = pos_p - pos_r  # (N, 3)
        env_change = self.env_encoder(delta_pos)  # (N, 64)
        
        # 4. 连接所有特征
        features = torch.cat([
            displacement,
            chem_change,
            env_change
        ], dim=-1)  # (N, 1 + chem_feat_dim + 64)
        
        # 5. 预测反应性得分
        reaction_scores = self.mlp(features).squeeze(-1)  # (N,)
        
        return reaction_scores


class CrossAttentionRP(nn.Module):
    """
    反应物和产物之间的交叉注意力
    
    实现R和P表示之间的信息交换，
    可选择使用注意力掩码聚焦于反应中心原子。
    
    参数：
        hidden_dim (int): 隐藏层维度（默认：256）
        num_heads (int): 注意力头数（默认：8）
        dropout (float): Dropout率（默认：0.1）
    """
    
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 交叉注意力：R关注P
        self.cross_attn_r2p = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 交叉注意力：P关注R
        self.cross_attn_p2r = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, feat_r, feat_p, center_scores=None):
        """
        双向交叉注意力的前向传播
        
        参数：
            feat_r (Tensor): 反应物特征，形状 (B, N, D)
            feat_p (Tensor): 产物特征，形状 (B, N, D)
            center_scores (Tensor, 可选): 反应中心得分，形状 (B, N)
                                         如果提供，用于掩码非中心原子
        
        返回：
            feat_r_out (Tensor): 增强后的反应物特征，形状 (B, N, D)
            feat_p_out (Tensor): 增强后的产物特征，形状 (B, N, D)
        """
        # R关注P
        if center_scores is not None:
            # 掩码旁观者原子（低反应得分）以聚焦反应中心
            attn_mask = (center_scores < 0.3)  # 阈值：0.3
            attn_r, _ = self.cross_attn_r2p(feat_r, feat_p, feat_p, key_padding_mask=attn_mask)
        else:
            attn_r, _ = self.cross_attn_r2p(feat_r, feat_p, feat_p)
        
        # 残差连接 & 归一化
        feat_r = self.norm(feat_r + attn_r)
        # 前馈网络
        feat_r = self.norm(feat_r + self.ffn(feat_r))
        
        # P关注R
        if center_scores is not None:
            attn_p, _ = self.cross_attn_p2r(feat_p, feat_r, feat_r, key_padding_mask=attn_mask)
        else:
            attn_p, _ = self.cross_attn_p2r(feat_p, feat_r, feat_r)
        
        # 残差连接 & 归一化
        feat_p = self.norm(feat_p + attn_p)
        # 前馈网络
        feat_p = self.norm(feat_p + self.ffn(feat_p))
        
        return feat_r, feat_p
