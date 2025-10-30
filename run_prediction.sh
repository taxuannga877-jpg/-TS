#!/bin/bash
# 运行预测脚本 - 为test_data_1的500个反应生成TS.xyz

echo "=========================================="
echo "🔮 过渡态结构预测"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查必需文件
echo "检查必需文件..."
if [ ! -f "outputs_xyz/best_model.pt" ]; then
    echo "❌ 错误：未找到模型文件 outputs_xyz/best_model.pt"
    exit 1
fi

if [ ! -d "test_data_1" ]; then
    echo "❌ 错误：未找到测试数据目录 test_data_1/"
    exit 1
fi

echo "✓ 模型文件: outputs_xyz/best_model.pt"
echo "✓ 测试数据: test_data_1/ ($(ls -d test_data_1/rxn* 2>/dev/null | wc -l) 个反应)"
echo ""

# 创建输出目录
OUTPUT_DIR="predictions"
echo "输出目录: $OUTPUT_DIR/"
mkdir -p "$OUTPUT_DIR"
echo ""

# 运行预测
echo "=========================================="
echo "开始预测..."
echo "=========================================="
echo ""

python predict_xyz.py \
  --test_dir ./test_data_1 \
  --model_path ./outputs_xyz/best_model.pt \
  --output_dir ./$OUTPUT_DIR \
  --reactant_file RS.xyz \
  --product_file PS.xyz

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 预测完成！"
    echo "=========================================="
    echo ""
    
    # 统计生成的文件
    ts_count=$(find "$OUTPUT_DIR" -name "TS.xyz" | wc -l)
    echo "生成的TS.xyz文件数: $ts_count"
    echo "输出位置: $OUTPUT_DIR/"
    echo ""
    
    # 显示示例
    if [ $ts_count -gt 0 ]; then
        echo "示例文件:"
        find "$OUTPUT_DIR" -name "TS.xyz" | head -5
        echo ""
    fi
    
    echo "💡 提示:"
    echo "  - 可以使用 get_rmsd.py 评估预测质量"
    echo "  - 命令: python get_rmsd.py $OUTPUT_DIR test_data_1"
else
    echo ""
    echo "=========================================="
    echo "❌ 预测失败"
    echo "=========================================="
    exit 1
fi

