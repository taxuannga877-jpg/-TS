"""
过渡态预测模型
策略：插值 + 微调偏移
ts_pos = alpha * r_pos + (1-alpha) * p_pos + delta_pos
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AtomEmbedding(nn.Module):
    """原子特征嵌入层"""
    
    def __init__(self, max_atomic_num=100, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(max_atomic_num, embed_dim)
    
    def forward(self, atomic_nums):
        """
        Args:
            atomic_nums: (N,) 原子序数
        Returns:
            atom_features: (N, embed_dim)
        """
        return self.embed(atomic_nums)


class DistanceFeatureExtractor(nn.Module):
    """距离矩阵特征提取"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 反应物距离 + 产物距离
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, r_dist, p_dist):
        """
        Args:
            r_dist: (N, N) 反应物距离矩阵
            p_dist: (N, N) 产物距离矩阵
        Returns:
            dist_features: (N, N, hidden_dim)
        """
        # 堆叠两个距离矩阵
        combined = torch.stack([r_dist, p_dist], dim=-1)  # (N, N, 2)
        features = self.mlp(combined)  # (N, N, hidden_dim)
        return features


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, input_dim, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
    def forward(self, x):
        """
        Args:
            x: (N, input_dim)
        Returns:
            encoded: (N, input_dim)
        """
        # 添加batch维度
        x = x.unsqueeze(0)  # (1, N, input_dim)
        encoded = self.transformer(x)
        return encoded.squeeze(0)  # (N, input_dim)


class TSPredictor(nn.Module):
    """
    过渡态结构预测模型
    
    输入：反应物坐标、产物坐标、原子类型
    输出：过渡态坐标
    
    策略：ts = alpha * r + (1-alpha) * p + delta
    """
    
    def __init__(self, 
                 atom_embed_dim=128,
                 dist_hidden_dim=256,
                 transformer_dim=512,
                 num_heads=8,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()
        
        # 原子嵌入
        self.atom_embedding = AtomEmbedding(max_atomic_num=100, embed_dim=atom_embed_dim)
        
        # 距离特征提取
        self.dist_extractor = DistanceFeatureExtractor(hidden_dim=dist_hidden_dim)
        
        # 特征融合
        self.feature_fusion = nn.Linear(
            atom_embed_dim + 6 + dist_hidden_dim,  # atom_embed + 坐标(r,p各3维) + dist
            transformer_dim
        )
        
        # Transformer编码器
        self.encoder = TransformerEncoder(
            input_dim=transformer_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 预测头
        self.alpha_head = nn.Sequential(
            nn.Linear(transformer_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 确保alpha在[0,1]
        )
        
        self.delta_head = nn.Sequential(
            nn.Linear(transformer_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3D偏移
        )
    
    def forward(self, atomic_nums, r_pos, p_pos, r_dist, p_dist):
        """
        前向传播
        
        Args:
            atomic_nums: (N,) 原子序数
            r_pos: (N, 3) 反应物坐标
            p_pos: (N, 3) 产物坐标
            r_dist: (N, N) 反应物距离矩阵
            p_dist: (N, N) 产物距离矩阵
            
        Returns:
            ts_pos_pred: (N, 3) 预测的过渡态坐标
            alpha: (N,) 插值系数
        """
        N = atomic_nums.size(0)
        
        # 1. 原子特征嵌入
        atom_features = self.atom_embedding(atomic_nums)  # (N, embed_dim)
        
        # 2. 距离特征（聚合到每个原子）
        dist_features = self.dist_extractor(r_dist, p_dist)  # (N, N, dist_hidden_dim)
        dist_features_pooled = dist_features.mean(dim=1)  # (N, dist_hidden_dim)
        
        # 3. 拼接所有特征
        combined_features = torch.cat([
            atom_features,          # (N, atom_embed_dim)
            r_pos,                  # (N, 3)
            p_pos,                  # (N, 3)
            dist_features_pooled    # (N, dist_hidden_dim)
        ], dim=-1)  # (N, total_dim)
        
        # 4. 特征融合
        fused_features = self.feature_fusion(combined_features)  # (N, transformer_dim)
        
        # 5. Transformer编码
        encoded = self.encoder(fused_features)  # (N, transformer_dim)
        
        # 6. 预测alpha和delta
        alpha = self.alpha_head(encoded).squeeze(-1)  # (N,)
        delta = self.delta_head(encoded)  # (N, 3)
        
        # 7. 计算过渡态坐标
        # ts = alpha * r + (1-alpha) * p + delta
        ts_pos_pred = alpha.unsqueeze(-1) * r_pos + \
                      (1 - alpha.unsqueeze(-1)) * p_pos + \
                      delta
        
        return ts_pos_pred, alpha


def create_model(config=None):
    """
    创建模型
    
    Args:
        config: 配置字典（可选）
        
    Returns:
        model: TSPredictor模型
    """
    if config is None:
        config = {
            'atom_embed_dim': 128,
            'dist_hidden_dim': 256,
            'transformer_dim': 512,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1
        }
    
    model = TSPredictor(**config)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    return model


