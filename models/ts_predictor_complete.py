"""
Reaction-Centric TS Predictor
专为TS坐标预测比赛设计的架构

核心创新：
1. 反应中心检测
2. R→P变化建模
3. 分层预测（反应中心 vs 旁观者）
4. 置信度预测
5. 多任务Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SchNet, radius_graph
from torch_geometric.nn import global_mean_pool


class ReactionCenterDetector(nn.Module):
    """反应中心检测器 - 识别参与反应的原子"""
    
    def __init__(self, chem_feat_dim=8, hidden_dim=128):
        super().__init__()
        
        # 输入：坐标位移 + 化学特征变化 + 局部环境变化
        input_dim = 1 + chem_feat_dim + 64
        
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
        
        # 局部环境编码
        self.env_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
    
    def forward(self, pos_r, pos_p, chem_feat_r, chem_feat_p, batch):
        """
        识别每个原子的反应性得分
        Args:
            pos_r, pos_p: 原子坐标
            chem_feat_r, chem_feat_p: 化学特征
            batch: 批次索引
        Returns:
            reaction_scores: (N,) 每个原子的反应性 [0, 1]
        """
        # 1. 坐标位移
        displacement = torch.norm(pos_p - pos_r, dim=-1, keepdim=True)  # (N, 1)
        
        # 2. 化学特征变化
        chem_change = chem_feat_p - chem_feat_r  # (N, 8)
        
        # 3. 局部几何变化
        delta_pos = pos_p - pos_r  # (N, 3)
        env_change = self.env_encoder(delta_pos)  # (N, 64)
        
        # 4. 拼接所有特征
        features = torch.cat([
            displacement,
            chem_change,
            env_change
        ], dim=-1)
        
        # 5. 预测反应性
        reaction_scores = self.mlp(features).squeeze(-1)  # (N,)
        
        return reaction_scores


class CrossAttentionRP(nn.Module):
    """R和P之间的交叉注意力（聚焦反应中心）"""
    
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.cross_attn_r2p = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_p2r = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, feat_r, feat_p, center_scores=None):
        """
        Args:
            feat_r, feat_p: (B, N, D)
            center_scores: (B, N) 反应性得分，用于加权attention
        """
        # R attend to P
        if center_scores is not None:
            # 只让反应中心参与复杂的attention
            attn_mask = (center_scores < 0.3)  # 旁观者原子mask掉
            attn_r, _ = self.cross_attn_r2p(feat_r, feat_p, feat_p, key_padding_mask=attn_mask)
        else:
            attn_r, _ = self.cross_attn_r2p(feat_r, feat_p, feat_p)
        
        feat_r = self.norm(feat_r + attn_r)
        feat_r = self.norm(feat_r + self.ffn(feat_r))
        
        # P attend to R
        if center_scores is not None:
            attn_p, _ = self.cross_attn_p2r(feat_p, feat_r, feat_r, key_padding_mask=attn_mask)
        else:
            attn_p, _ = self.cross_attn_p2r(feat_p, feat_r, feat_r)
        
        feat_p = self.norm(feat_p + attn_p)
        feat_p = self.norm(feat_p + self.ffn(feat_p))
        
        return feat_r, feat_p


class ChemicalFeatureEncoder(nn.Module):
    """RDKit化学特征编码器"""
    
    def __init__(self, chem_feat_dim=8, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(chem_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, chem_feat):
        return self.encoder(chem_feat)


class ReactionCentricTSPredictor(nn.Module):
    """
    反应中心为核心的TS预测器
    - 自动识别反应中心
    - 对反应中心和旁观者分别建模
    - 预测置信度
    """
    
    def __init__(self,
                 schnet_hidden=256,
                 schnet_filters=256,
                 schnet_interactions=6,
                 schnet_gaussians=100,
                 chem_feat_dim=8,
                 chem_hidden=128,
                 cutoff=6.0,
                 dropout=0.15):
        super().__init__()
        
        print("🔥 创建Reaction-Centric TS Predictor")
        print("   - 反应中心检测")
        print("   - R↔P Cross-Attention")
        print("   - 分层预测（反应中心 vs 旁观者）")
        print("   - 置信度预测")
        
        # 1. 化学特征编码
        self.chem_encoder = ChemicalFeatureEncoder(chem_feat_dim, chem_hidden)
        
        # 2. 反应中心检测器
        self.reaction_center_detector = ReactionCenterDetector(
            chem_feat_dim=chem_feat_dim,
            hidden_dim=128
        )
        
        # 3. 几何编码器（单一SchNet）
        self.schnet = SchNet(
            hidden_channels=schnet_hidden,
            num_filters=schnet_filters,
            num_interactions=schnet_interactions,
            num_gaussians=schnet_gaussians,
            cutoff=cutoff,
            readout='add'
        )
        
        # 4. R→P变化编码
        self.change_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 5. Cross-Attention
        self.cross_attention = CrossAttentionRP(
            hidden_dim=schnet_hidden,
            num_heads=8,
            dropout=dropout
        )
        
        # 6. 特征融合
        fusion_dim = chem_hidden + 2 * schnet_hidden + 64
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, schnet_hidden * 2),
            nn.LayerNorm(schnet_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(schnet_hidden * 2, schnet_hidden)
        )
        
        # 7. alpha预测
        self.alpha_head = nn.Sequential(
            nn.Linear(schnet_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 8. 分层预测器
        # 反应中心：复杂预测
        self.center_predictor = nn.Sequential(
            nn.Linear(schnet_hidden, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        
        # 旁观者：简单修正
        self.spectator_predictor = nn.Sequential(
            nn.Linear(schnet_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # 9. 置信度预测
        self.confidence_head = nn.Sequential(
            nn.Linear(schnet_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def encode_geometry(self, z, pos, batch):
        """使用SchNet提取节点特征"""
        h = self.schnet.embedding(z)
        edge_index = radius_graph(pos, r=self.schnet.cutoff, batch=batch)
        edge_weight = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
        edge_attr = self.schnet.distance_expansion(edge_weight)
        
        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        
        return h
    
    def forward(self, data_r, data_p):
        batch_size = data_r.batch.max().item() + 1
        
        # 1. 化学特征
        chem_feat_r = self.chem_encoder(data_r.chem_feat)
        chem_feat_p = self.chem_encoder(data_p.chem_feat)
        chem_feat = (chem_feat_r + chem_feat_p) / 2
        
        # 2. 反应中心检测
        reaction_scores = self.reaction_center_detector(
            data_r.pos, data_p.pos,
            data_r.chem_feat, data_p.chem_feat,
            data_r.batch
        )  # (N,) 每个原子的反应性
        
        # 3. 几何特征
        geom_feat_r = self.encode_geometry(data_r.z, data_r.pos, data_r.batch)
        geom_feat_p = self.encode_geometry(data_p.z, data_p.pos, data_p.batch)
        
        # 4. R→P变化特征
        delta_pos = data_p.pos - data_r.pos
        change_feat = self.change_encoder(delta_pos)
        
        # 5. 转换为batch格式用于Cross-Attention
        max_atoms = max([(data_r.batch == i).sum() for i in range(batch_size)])
        
        geom_r_batch = torch.zeros(batch_size, max_atoms, geom_feat_r.size(-1), device=geom_feat_r.device)
        geom_p_batch = torch.zeros(batch_size, max_atoms, geom_feat_p.size(-1), device=geom_feat_p.device)
        scores_batch = torch.zeros(batch_size, max_atoms, device=reaction_scores.device)
        
        for i in range(batch_size):
            mask = (data_r.batch == i)
            n = mask.sum()
            geom_r_batch[i, :n] = geom_feat_r[mask]
            geom_p_batch[i, :n] = geom_feat_p[mask]
            scores_batch[i, :n] = reaction_scores[mask]
        
        # 6. Cross-Attention（聚焦反应中心）
        geom_r_enhanced, geom_p_enhanced = self.cross_attention(
            geom_r_batch, geom_p_batch, scores_batch
        )
        
        # 7. 展平
        geom_r_flat = []
        geom_p_flat = []
        for i in range(batch_size):
            n = (data_r.batch == i).sum()
            geom_r_flat.append(geom_r_enhanced[i, :n])
            geom_p_flat.append(geom_p_enhanced[i, :n])
        
        geom_feat_r = torch.cat(geom_r_flat, dim=0)
        geom_feat_p = torch.cat(geom_p_flat, dim=0)
        
        # 8. 融合所有特征
        fused_feat = torch.cat([
            chem_feat,
            geom_feat_r,
            geom_feat_p,
            change_feat
        ], dim=-1)
        
        fused_feat = self.feature_fusion(fused_feat)
        
        # 9. 预测alpha
        alpha = self.alpha_head(fused_feat)  # (N, 1)
        
        # 10. 分层预测delta
        # 反应中心：精细预测
        delta_center = self.center_predictor(fused_feat)
        
        # 旁观者：简单修正
        delta_spectator = self.spectator_predictor(fused_feat)
        
        # 根据反应性得分混合
        reaction_scores_3d = reaction_scores.unsqueeze(-1)  # (N, 1)
        delta = (reaction_scores_3d * delta_center + 
                 (1 - reaction_scores_3d) * delta_spectator)
        
        # 11. 最终TS坐标
        ts_pred = alpha * data_r.pos + (1 - alpha) * data_p.pos + delta
        
        # 12. 置信度预测
        confidence = self.confidence_head(fused_feat)  # (N, 1)
        
        return ts_pred, alpha, delta, confidence, reaction_scores


class ReactionCentricLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self,
                 coord_weight=1.0,
                 dist_weight=0.2,
                 center_weight=0.3,
                 confidence_weight=0.1,
                 collision_weight=0.1):
        super().__init__()
        self.coord_weight = coord_weight
        self.dist_weight = dist_weight
        self.center_weight = center_weight
        self.confidence_weight = confidence_weight
        self.collision_weight = collision_weight
    
    def forward(self, pred_ts, true_ts, alpha, confidence, reaction_scores, 
                data_r, data_p, batch):
        """
        Args:
            pred_ts, true_ts: 预测和真实TS坐标
            alpha: 插值参数
            confidence: 置信度预测
            reaction_scores: 反应中心得分
            data_r, data_p: R和P的数据
            batch: 批次索引
        """
        # 1. 坐标Loss（反应中心加权）
        coord_diff = (pred_ts - true_ts) ** 2
        # 反应中心的误差权重更大
        weights = 1.0 + 2.0 * reaction_scores.unsqueeze(-1)
        loss_coord = (coord_diff * weights).mean()
        
        # 2. 距离保持Loss
        batch_size = batch.max().item() + 1
        loss_dist = 0.0
        loss_collision = 0.0
        
        for i in range(batch_size):
            mask = (batch == i)
            pred_mol = pred_ts[mask]
            true_mol = true_ts[mask]
            
            # 距离矩阵
            pred_dist = torch.cdist(pred_mol, pred_mol)
            true_dist = torch.cdist(true_mol, true_mol)
            loss_dist += F.mse_loss(pred_dist, true_dist)
            
            # 碰撞惩罚
            too_close = torch.relu(0.8 - pred_dist)
            mask_diag = torch.eye(pred_dist.size(0), device=pred_dist.device).bool()
            too_close = too_close.masked_fill(mask_diag, 0)
            loss_collision += too_close.mean()
        
        loss_dist /= batch_size
        loss_collision /= batch_size
        
        # 3. 反应中心识别Loss
        # 真实反应中心：位移 > 0.5Å 的原子
        true_displacement = torch.norm(true_ts - data_r.pos, dim=-1)
        true_center = (true_displacement > 0.5).float()
        with torch.cuda.amp.autocast(enabled=False):
            loss_center = F.binary_cross_entropy(reaction_scores.float(), true_center.float())
        
        # 4. 置信度Loss
        # 预测RMSD，置信度应该与(1 / (1 + RMSD))相关
        with torch.no_grad():
            atom_errors = torch.norm(pred_ts - true_ts, dim=-1)
            target_conf = 1.0 / (1.0 + atom_errors)
        loss_confidence = F.mse_loss(confidence.squeeze(-1), target_conf)
        
        # 5. Alpha约束
        loss_alpha = torch.mean(
            torch.relu(0.05 - alpha) + torch.relu(alpha - 0.95)
        )
        
        # 总损失
        total_loss = (self.coord_weight * loss_coord +
                      self.dist_weight * loss_dist +
                      self.center_weight * loss_center +
                      self.confidence_weight * loss_confidence +
                      self.collision_weight * loss_collision +
                      0.01 * loss_alpha)
        
        return total_loss, {
            'coord': loss_coord.item(),
            'dist': loss_dist.item(),
            'center': loss_center.item(),
            'confidence': loss_confidence.item(),
            'collision': loss_collision.item(),
            'alpha': loss_alpha.item()
        }


def create_reaction_centric_model(config=None):
    """创建Reaction-Centric模型"""
    if config is None:
        config = {}
    
    model = ReactionCentricTSPredictor(
        schnet_hidden=config.get('schnet_hidden', 256),
        schnet_filters=config.get('schnet_filters', 256),
        schnet_interactions=config.get('schnet_interactions', 6),
        schnet_gaussians=config.get('schnet_gaussians', 100),
        chem_feat_dim=config.get('chem_feat_dim', 8),
        chem_hidden=config.get('chem_hidden', 128),
        cutoff=config.get('cutoff', 6.0),
        dropout=config.get('dropout', 0.15)
    )
    
    return model

