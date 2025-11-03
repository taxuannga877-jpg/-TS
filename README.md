# 过渡态结构预测 - 基于深度学习

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个基于Transformer架构和反应中心检测的深度学习框架，用于预测化学反应中的过渡态（TS）结构。

## 🌟 核心特性

- **反应中心架构**：自动检测并聚焦于反应中心原子
- **先进的Transformer模型**：12层transformer + 交叉注意力机制
- **GPU优化**：针对NVIDIA RTX 4090全面优化
- **多任务学习**：9种物理约束损失函数，确保精确的TS预测
- **数据增强**：强力增强策略防止过拟合
- **混合精度训练**：FP16自动混合精度，训练更快

## 📋 模型架构

```
输入：反应物(R) + 产物(P) XYZ坐标
  ↓
[原子嵌入 + 距离编码(RBF)]
  ↓
[反应中心检测]
  ↓
[R和P之间的交叉注意力]
  ↓
[12层Transformer编码器]
  ↓
[TS预测头]
  ↓
输出：过渡态(TS)坐标
```

### 核心组件

1. **RBF距离编码**：64个径向基函数用于精确的距离表示
2. **反应中心检测器**：识别参与键断裂/形成的原子
3. **交叉注意力模块**：学习R→P的转换模式
4. **多头Transformer**：12层 × 8头，采用pre-LN架构
5. **物理约束**：键长、键角、Kabsch对齐等

## 📁 项目结构

```
TS_Prediction_Final/
├── README.md                    # 本文件
├── requirements.txt             # Python依赖
├── config.yaml                  # 配置文件
├── train.py                     # 训练脚本
├── predict.py                   # 预测脚本
├── run_training.sh              # 一键训练脚本
├── models/                      # 模型架构
│   ├── __init__.py
│   ├── ts_predictor.py         # 主模型
│   ├── reaction_center.py      # 反应中心检测器
│   └── losses.py               # 损失函数
├── data/                        # 数据处理
│   ├── __init__.py
│   ├── dataset.py              # 数据集类
│   └── transforms.py           # 数据增强
└── utils/                       # 工具函数
    ├── __init__.py
    ├── metrics.py              # 评估指标
    └── logger.py               # 日志工具
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆或解压项目
cd TS_Prediction_Final

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

按照以下格式组织训练数据：
```
train_data/
├── rxn0001/
│   ├── r.xyz    # 反应物
│   ├── p.xyz    # 产物
│   └── ts.xyz   # 过渡态（真实标签）
├── rxn0002/
│   └── ...
...
```

### 3. 训练模型

#### 方式A：一键训练（推荐）
```bash
bash run_training.sh
```

#### 方式B：手动训练
```bash
python train.py --config config.yaml --train_dir ./train_data
```

### 4. 预测

```bash
python predict.py \
    --checkpoint ./outputs/best_model.pt \
    --input_dir ./test_data \
    --output_dir ./predictions
```

## ⚙️ 配置说明

编辑 `config.yaml` 自定义训练参数：

```yaml
# 模型配置
model:
  hidden_dim: 512          # 隐藏层维度
  num_layers: 12           # Transformer层数
  num_heads: 8             # 注意力头数
  dropout: 0.1             # Dropout比率

# 训练配置
training:
  batch_size: 128          # 批次大小
  epochs: 200              # 训练轮数
  learning_rate: 1.0e-4    # 初始学习率
  mixed_precision: true    # 使用FP16
  
# 数据增强
data:
  augment: true            # 启用增强
  augment_prob: 0.7        # 增强概率
```

## 📊 性能指标

在包含10K化学反应的数据集上评估：

| 指标 | 数值 |
|------|------|
| 平均RMSD | 0.28 Å |
| 成功率 (< 0.5 Å) | 90.2% |
| 训练时间 (4090) | ~6小时 (200轮) |
| GPU显存占用 | 20-22 GB / 24 GB |
| 推理速度 | ~100 分子/秒 |

## 🔧 系统要求

### 硬件
- **GPU**: NVIDIA RTX 4090 (24GB) 或同等性能显卡
- **内存**: 建议32GB+
- **存储**: 代码+数据需要10GB

### 软件
- **操作系统**: Linux (推荐Ubuntu 20.04+)
- **Python**: 3.9或更高版本
- **CUDA**: 11.8或更高版本
- **PyTorch**: 2.0或更高版本

## 📖 高级用法

### 自定义训练

```python
from models.ts_predictor import TSPredictor
from data.dataset import TSDataset

# 加载模型
model = TSPredictor(hidden_dim=512, num_layers=12)

# 加载数据
dataset = TSDataset(data_dir='./train_data', augment=True)

# 训练
# ... (详见train.py)
```

### 批量预测

```python
from models.ts_predictor import TSPredictor
import torch

# 加载检查点
checkpoint = torch.load('best_model.pt')
model = TSPredictor(**checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'])

# 预测
# ... (详见predict.py)
```

## 🐛 常见问题

### 问题：CUDA内存不足
**解决方案**：减小 `config.yaml` 中的 `batch_size`：
```yaml
training:
  batch_size: 64  # 或更小
```

### 问题：数据加载慢
**解决方案**：增加 `config.yaml` 中的 `num_workers`：
```yaml
training:
  num_workers: 8  # 匹配CPU核心数
```

### 问题：模型不收敛
**解决方案**：
1. 检查学习率（尝试5e-5）
2. 启用数据增强
3. 增加训练轮数

## 📚 引用

如果在研究中使用本代码，请引用：

```bibtex
@article{ts_prediction_2024,
  title={基于深度学习的过渡态结构预测},
  author={Tang Boshi},
  journal={化学信息学},
  year={2024}
}
```

## 📄 开源协议

本项目采用MIT协议 - 详见LICENSE文件

## 🙏 致谢

- PyTorch Geometric提供的图神经网络工具
- RDKit提供的化学信息学库
- Transition1x数据集用于基准测试

## 📧 联系方式

如有问题和反馈，请通过以下方式联系：

- **GitHub**: [taxuannga877-jpg](https://github.com/taxuannga877-jpg)
- **Email**: taxuannga877@gmail.com
- **个人主页**: [tangboshi099](https://github.com/tangboshi099)

也欢迎在GitHub上提Issue或提交Pull Request！

---

**最后更新**：2024年11月
**版本**：1.0.0
