# 快速开始指南

## 🚀 3分钟上手

```bash
# 1. 进入项目目录
cd TS_Prediction_Final

# 2. 安装依赖（首次使用）
pip install -r requirements.txt

# 3. 准备训练数据
# 将数据放在 train_data/ 目录，格式：
# train_data/rxn0001/{r.xyz, p.xyz, ts.xyz}
# train_data/rxn0002/{r.xyz, p.xyz, ts.xyz}
# ...

# 4. 一键启动训练
bash run_training.sh

# 5. 预测（训练完成后）
python predict.py \
    --checkpoint outputs/run_*/best_model.pt \
    --input_dir test_data \
    --output_dir predictions
```

---

## 📂 数据格式

### XYZ文件格式示例：

```
9
Properties=species:S:1:pos:R:3 pbc="F F F"
C       -1.31180026       0.00791080       0.03195078
C        0.17396933      -0.03116598       0.02107604
N        0.90716467      -0.90183274       0.64560492
O        2.17285288      -0.54960924       0.34537693
...
```

### 目录结构：

```
train_data/
├── rxn0001/
│   ├── r.xyz      # 反应物 (Reactant)
│   ├── p.xyz      # 产物 (Product)
│   └── ts.xyz     # 过渡态 (Transition State, 真实标签)
├── rxn0002/
│   └── ...
...
```

---

## ⚙️ 配置调整

编辑 `config.yaml` 进行自定义配置：

```yaml
# 重要参数
training:
  batch_size: 128      # 批次大小（根据GPU调整）
  epochs: 200          # 训练轮数
  learning_rate: 1.0e-4  # 学习率
  mixed_precision: true  # 混合精度（推荐开启）

model:
  hidden_dim: 512      # 模型维度
  num_layers: 12       # Transformer层数
```

---

## 🔍 监控训练

### 方法1: 查看日志
```bash
tail -f logs/training_*.log
```

### 方法2: TensorBoard
```bash
tensorboard --logdir outputs/run_*/logs --port 6006
# 打开浏览器访问 http://localhost:6006
```

### 方法3: GPU监控
```bash
watch -n 1 nvidia-smi
```

---

## 🐛 常见问题

### Q1: CUDA内存不足
**解决方法：**
```yaml
# 编辑 config.yaml
training:
  batch_size: 64  # 减小批次
```

### Q2: 依赖安装失败
**解决方法：**
```bash
# 使用conda安装RDKit
conda install -c conda-forge rdkit

# 或使用pip
pip install rdkit-pypi
```

### Q3: 数据加载错误
**检查：**
1. XYZ文件格式是否正确
2. 目录结构是否符合要求
3. 文件名是否正确 (r.xyz, p.xyz, ts.xyz)

---

## 📊 预期结果

**训练时间 (RTX 4090):**
- 100轮: ~3小时
- 200轮: ~6小时

**性能指标:**
- RMSD: < 0.30 Å
- 成功率: > 85%

**输出文件:**
```
outputs/run_20241104_HHMMSS/
├── config.yaml              # 训练配置
├── best_model.pt            # 最佳模型
├── checkpoint_epoch_*.pt    # 定期检查点
└── logs/                    # TensorBoard日志
```

---

## 📧 获取帮助

1. 查看完整文档: `README.md`
2. 查看代码注释: 所有函数都有详细docstring
3. 联系作者: taxuannga877@gmail.com
4. GitHub Issues: [提交问题](https://github.com/taxuannga877-jpg)

---

**祝训练顺利！** 🎉
