#!/bin/bash
# train_colab.sh - 一键Colab训练脚本

echo "========================================="
echo "🚀 胸部X光疾病分类 - Colab训练脚本"
echo "========================================="

# 1. 检查是否在Colab环境
if python -c "import sys; print('Colab' if 'google.colab' in sys.modules else 'Not Colab')" | grep -q "Not Colab"; then
    echo "⚠ 警告：不在Colab环境中"
    echo "请将本脚本上传到Colab运行以获得GPU加速"
    exit 1
fi

# 2. 设置项目变量
PROJECT_NAME="chest_xray_project"
LOCAL_PROJECT_PATH="/content/$PROJECT_NAME"  # 假设项目已上传到此位置

# 3. 安装基础依赖
echo "📦 安装依赖..."
pip install -q torch torchvision torchaudio
pip install -q efficientnet-pytorch albumentations
pip install -q pandas scikit-learn pyyaml tqdm

# 4. 运行训练
echo "🚀 开始训练..."
cd "$LOCAL_PROJECT_PATH"
python main.py \
    --mode train \
    --colab \
    --epochs 50 \
    --batch-size 32

# 5. 保存结果到Google Drive
echo "💾 保存结果到Google Drive..."
from google.colab import drive
drive.mount('/content/drive')

# 创建备份目录
DRIVE_BACKUP="/content/drive/MyDrive/Colab_Backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DRIVE_BACKUP"

# 复制重要文件
cp -r outputs/ "$DRIVE_BACKUP/"
cp config/config.yaml "$DRIVE_BACKUP/"
cp logs/*.log "$DRIVE_BACKUP/" 2>/dev/null || true

echo "✅ 训练完成！结果已保存到: $DRIVE_BACKUP"