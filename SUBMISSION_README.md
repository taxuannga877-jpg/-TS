# 化学反应过渡态结构预测 - 参赛作品

## 📦 文件结构

```
111aaa_TS/
├── ts_prediction/              # 核心代码模块
│   ├── data/                   # 数据处理
│   │   ├── xyz_dataset.py      # XYZ格式数据集加载
│   │   └── __init__.py
│   ├── models/                 # 模型定义
│   │   ├── ts_predictor.py     # 过渡态预测模型
│   │   └── __init__.py
│   └── utils/                  # 工具函数
│       ├── metrics.py          # RMSD计算
│       └── xyz_io.py           # XYZ文件读写
├── outputs_xyz/                # 训练输出
│   └── best_model.pt          # 训练好的最佳模型 ⭐
├── train_data/                 # 训练数据（10073个反应）
├── test_data_1/                # 测试数据（500个反应）
├── train_xyz.py               # 训练脚本 ⭐
├── predict_xyz.py             # 预测脚本 ⭐
├── requirements.txt           # Python依赖 ⭐
├── get_rmsd.py                # RMSD评估脚本
├── README.md                  # 项目说明
├── 赛题规则.pdf               # 比赛规则
└── 赛题数据说明.docx          # 数据说明

⭐ = 比赛必需文件
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# Python 3.8+
# PyTorch 2.0+
# CPU或GPU均可运行
```

### 2. 模型训练（可选）

如果需要重新训练模型：

```bash
python train_xyz.py \
  --train_dir ./train_data \
  --output_dir ./outputs_xyz \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --num_workers 8
```

**训练参数说明**：
- `train_dir`: 训练数据目录（包含rxnX子文件夹，每个子文件夹包含RS.xyz, PS.xyz, TS.xyz）
- `output_dir`: 输出目录（保存模型和日志）
- `batch_size`: 批大小（根据内存调整）
- `epochs`: 训练轮数
- `lr`: 学习率
- `num_workers`: 数据加载线程数

**训练输出**：
- `best_model.pt`: 验证集上RMSD最低的模型
- `checkpoint_epoch_*.pt`: 定期保存的检查点
- `logs/`: TensorBoard日志

### 3. 预测测试集（重要）

使用训练好的模型预测测试集的过渡态结构：

```bash
python predict_xyz.py \
  --test_dir ./test_data_1 \
  --model_path ./outputs_xyz/best_model.pt \
  --output_dir ./predictions
```

**预测参数说明**：
- `test_dir`: 测试数据目录（包含rxn1-rxn500子文件夹）
- `model_path`: 训练好的模型路径
- `output_dir`: 输出目录（将为每个反应创建rxnX/TS.xyz）
- `--reactant_file`: 反应物文件名（默认RS.xyz）
- `--product_file`: 产物文件名（默认PS.xyz）

**预测输出**：
```
predictions/
├── rxn1/
│   └── TS.xyz    # 预测的过渡态结构
├── rxn2/
│   └── TS.xyz
...
└── rxn500/
    └── TS.xyz
```

### 4. 评估结果（可选）

如果有真实的TS.xyz标签，可以评估预测结果：

```bash
python get_rmsd.py ./predictions ./test_data_1
```

## 🧠 模型架构

### 核心思想

过渡态结构位于反应物和产物之间的能量曲面上，我们采用**端到端的深度学习模型**直接从反应物和产物的3D结构预测过渡态结构。

### 模型设计

**输入**：
- 反应物原子坐标 (N×3)
- 产物原子坐标 (N×3)
- 原子类型 (N,)
- 距离矩阵 (N×N)

**模型组件**：

1. **原子嵌入层** (AtomEmbedding)
   - 将原子序数映射到高维特征空间（128维）
   - 使用可学习的嵌入表征不同元素的化学性质

2. **距离特征提取器** (DistanceFeatureExtractor)
   - 提取反应物和产物的距离矩阵特征
   - 使用MLP编码原子间距离信息
   - 捕捉分子的几何结构特征

3. **Transformer编码器** (TransformerEncoder)
   - 4层多头注意力机制（8个头）
   - 学习原子间的长程依赖关系
   - 维度：256 → 512 → 256
   - Dropout: 0.1（防止过拟合）

4. **过渡态预测头** (TSPredictor)
   - 采用**插值策略**：TS = α·R + (1-α)·P + Δ
   - α: 可学习的插值权重
   - Δ: 残差位移修正
   - 输出过渡态坐标和α权重

**输出**：
- 过渡态原子坐标 (N×3)
- 插值权重 α ∈ [0,1]

### 训练策略

**损失函数**：
- 主损失：MSE（均方误差）
- 优化目标：最小化预测TS与真实TS的坐标差异

**优化器**：
- AdamW（weight_decay=1e-5）
- 学习率：1e-4
- 学习率调度：CosineAnnealingLR

**数据增强**：
- 随机旋转（保持分子几何不变性）
- 90%训练集 / 10%验证集划分

**评估指标**：
- RMSD（Root Mean Square Deviation）：衡量预测结构与真实结构的偏差
- Success Rate：RMSD < 0.5Å的样本比例

## 📊 训练结果

**最佳模型性能**（Epoch 4）：
- 验证集RMSD: **0.3700 Å**
- Success Rate: **78.27%**
- 模型参数量: 13,020,932

**训练曲线**：
| Epoch | Train RMSD | Val RMSD | Success Rate |
|-------|-----------|----------|--------------|
| 1 | 0.4131 Å | 0.4007 Å | 73.12% |
| 2 | 0.4006 Å | 0.3937 Å | 74.50% |
| 3 | 0.3850 Å | 0.3771 Å | 77.68% |
| **4** | **0.3751 Å** | **0.3700 Å** | **78.27%** ⭐ |
| 5 | 0.3687 Å | 0.3716 Å | 78.37% |

## 🔬 技术特点

### 1. 模块化设计
- 清晰的代码结构（数据/模型/工具分离）
- 易于扩展和修改
- 完整的注释和文档字符串

### 2. 依赖管理
- 提供完整的`requirements.txt`
- 所有依赖均为常用库，易于安装
- 支持CPU和GPU运行

### 3. 可扩展性
- 支持命令行参数配置
- 灵活的模型超参数调整
- 易于替换模型架构

### 4. 结果复现
- 固定随机种子（seed=42）
- 详细的训练日志
- 完整的模型检查点

## 📝 依赖列表

```
torch>=2.0.0          # 深度学习框架
numpy>=1.20.0         # 数值计算
ase>=3.22.0           # 分子结构处理
scipy>=1.7.0          # 科学计算（RMSD对齐）
tqdm>=4.62.0          # 进度条
tensorboard>=2.10.0   # 训练可视化
```

## 💡 创新点

1. **端到端学习**：直接从3D结构预测，无需手工特征工程
2. **插值策略**：结合反应物和产物的几何信息
3. **注意力机制**：捕捉原子间的长程相互作用
4. **残差学习**：在插值基础上学习精细修正

## 🎯 未来改进方向

1. **预训练模型**：在更大规模分子数据集上预训练
2. **图神经网络**：显式建模化学键和分子图结构
3. **物理约束**：引入能量、力场等物理约束
4. **集成学习**：训练多个模型进行ensemble预测
5. **主动学习**：选择不确定样本进行标注

## 📧 联系方式

- 作者：[您的姓名]
- 邮箱：[您的邮箱]
- 日期：2025年10月30日

---

*本项目遵循比赛规则，所有代码均为原创实现。*

