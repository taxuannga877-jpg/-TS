# 🧪 化学反应过渡态结构预测

基于深度学习的化学反应过渡态（Transition State）结构预测系统。使用Transformer架构预测反应物和产物之间的过渡态原子坐标。

## 🎯 项目简介

本项目实现了一个端到端的过渡态预测模型，能够：
- 从反应物（RS）和产物（PS）的XYZ文件预测过渡态（TS）结构
- 使用等变Transformer网络处理分子几何信息
- 在验证集上达到 **RMSD = 0.333 Å**，成功率 **82.74%** (RMSD < 0.5Å)

## 📊 模型性能

| 指标 | 数值 |
|------|------|
| 验证集RMSD | 0.333 Å |
| 成功率 (RMSD<0.5Å) | 82.74% |
| 训练集大小 | 9,065 个反应 |
| 验证集大小 | 1,008 个反应 |
| 模型参数量 | 13M |

**训练趋势**: RMSD从0.399Å降至0.333Å (16.5%改善)

## 🏗️ 模型架构

```
输入层: 原子编号 + 反应物/产物坐标 + 距离矩阵
   ↓
原子嵌入 (128维) + 距离特征提取
   ↓
特征融合 (256维)
   ↓
Transformer Encoder (6层, 8头)
   ↓
过渡态预测层
   ↓
输出: 过渡态原子坐标
```

**核心特性**:
- ✅ 基于Transformer的序列建模
- ✅ 距离矩阵作为几何特征
- ✅ 插值策略：TS = α·RS + (1-α)·PS + Δ
- ✅ 余弦退火学习率调度

## 📁 项目结构

```
.
├── ts_prediction/              # 核心代码包
│   ├── models/                 # 模型定义
│   │   ├── ts_predictor.py     # Transformer预测器
│   │   └── __init__.py
│   ├── data/                   # 数据加载
│   │   ├── xyz_dataset.py      # XYZ数据集
│   │   └── __init__.py
│   └── utils/                  # 工具函数
│       ├── xyz_io.py           # XYZ文件读写
│       └── metrics.py          # RMSD计算
├── train_xyz.py                # 训练脚本
├── predict_xyz.py              # 预测脚本
├── requirements.txt            # 依赖包
├── get_rmsd.py                 # RMSD评估脚本
└── README.md                   # 项目文档
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/ts-prediction.git
cd ts-prediction

# 安装依赖
pip install -r requirements.txt
```

**依赖包**:
- Python >= 3.8
- PyTorch >= 2.0
- ASE (Atomic Simulation Environment)
- NumPy, SciPy
- tqdm, tensorboard

### 2. 数据准备

数据格式：
```
train_data/
├── rxn1/
│   ├── RS.xyz    # 反应物
│   ├── PS.xyz    # 产物
│   └── TS.xyz    # 过渡态 (训练标签)
├── rxn2/
│   └── ...
└── ...
```

### 3. 训练模型

```bash
python train_xyz.py \
    --train_dir ./train_data \
    --output_dir ./outputs_xyz \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4
```

**训练参数**:
- `--hidden_dim 256`: Transformer隐藏层维度
- `--num_layers 6`: Transformer层数
- `--num_heads 8`: 注意力头数
- `--dropout 0.1`: Dropout率

**性能优化**:
- CPU训练：自动使用所有核心
- GPU训练：支持CUDA加速（需要兼容的GPU）
- 自动保存最佳模型和checkpoint

### 4. 预测过渡态

```bash
python predict_xyz.py \
    --test_dir ./test_data_1 \
    --model_path ./outputs_xyz/best_model.pt \
    --output_dir ./test_data_1
```

输出: 每个反应文件夹中生成 `TS_pred.xyz`

### 5. 评估结果

```bash
python get_rmsd.py test_data_1 test_data_1
```

## 📈 训练监控

使用TensorBoard查看训练过程：

```bash
tensorboard --logdir outputs_xyz/logs
```

监控指标：
- `Loss/train`, `Loss/val`: 训练/验证损失
- `RMSD/train`, `RMSD/val`: RMSD指标
- `SuccessRate/val`: 成功率 (RMSD<0.5Å)
- `LR`: 学习率变化

## 🔬 技术细节

### 损失函数
```python
Loss = MSE(predicted_coords, true_coords)
```

### RMSD计算
使用Kabsch算法对齐后计算均方根偏差：
```python
RMSD = sqrt(mean(||predicted - true||²))
```

### 训练策略
1. **优化器**: AdamW (weight_decay=1e-5)
2. **学习率调度**: CosineAnnealingLR (T_max=50)
3. **Early Stopping**: 监控验证RMSD
4. **Checkpoint**: 每轮保存最佳模型

## 📝 实验结果

训练50轮后的性能：

| Epoch | Train RMSD | Val RMSD | Success Rate |
|-------|------------|----------|--------------|
| 1     | 0.411 Å    | 0.399 Å  | 73.61%       |
| 10    | 0.348 Å    | 0.352 Å  | 80.26%       |
| 29    | 0.251 Å    | 0.333 Å  | 83.43%       |
| 42    | 0.216 Å    | 0.333 Å  | 82.74% ⭐    |
| 50    | 0.211 Å    | 0.334 Å  | 82.84%       |

## 🛠️ 高级用法

### 自定义模型架构

编辑 `ts_prediction/models/ts_predictor.py`:

```python
model = create_model(
    hidden_dim=512,      # 增加模型容量
    num_layers=8,        # 更深的网络
    num_heads=16,        # 更多注意力头
    dropout=0.2          # 更强的正则化
)
```

### 数据增强

在训练时可以添加：
- 分子旋转/平移
- 原子顺序随机化
- 噪声注入

### 模型集成

训练多个模型并平均预测：
```python
pred = (model1(x) + model2(x) + model3(x)) / 3
```

## 💡 注意事项

1. **GPU兼容性**: 需要CUDA compute capability >= sm_50
   - 对于新GPU (如RTX 40/50系列)，需要PyTorch Nightly版本
   - CPU训练已优化，性能良好

2. **内存需求**: 
   - 训练: ~8GB RAM (batch_size=16)
   - 推理: ~2GB RAM

3. **数据质量**:
   - 确保RS/PS/TS的原子顺序一致
   - 坐标单位为Ångström (Å)

## 📚 参考资料

- [Transformer论文](https://arxiv.org/abs/1706.03762)
- [ASE文档](https://wiki.fysik.dtu.dk/ase/)
- [PyTorch几何深度学习](https://pytorch-geometric.readthedocs.io/)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 👨‍💻 作者

化学反应过渡态预测竞赛参赛项目

---

⭐ 如果这个项目对你有帮助，请给个Star！
