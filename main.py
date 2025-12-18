import argparse
import os
import sys
import yaml
from pathlib import Path

def setup_environment(is_colab=False):
    """设置环境，导入必要的包"""
    if is_colab:
        # 在Colab中可能需要安装额外的包
        print("Running in Google Colab environment")
        !pip install -q torch torchvision torchtext torchaudio
        !pip install -q efficientnet-pytorch
        !pip install -q albumentations
    
    # 添加项目根目录到路径
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))

def main():
    parser = argparse.ArgumentParser(description='胸部X光疾病分类')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'predict'], 
                       default='train', help='运行模式')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--colab', action='store_true', help='是否在Colab中运行')
    parser.add_argument('--data_dir', type=str, help='数据目录')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 根据环境选择路径
    if args.colab:
        config['paths'] = config['paths_colab']
        # 挂载Google Drive
        from google.colab import drive
        drive.mount(config['paths']['drive_mount'])
    else:
        config['paths'] = config['paths_local']
    
    # 覆盖命令行参数
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    
    # 创建输出目录
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    # 运行相应模式
    if args.mode == 'train':
        from training.train import train_model
        train_model(config)
    elif args.mode == 'test':
        from training.evaluate import evaluate_model
        evaluate_model(config)
    elif args.mode == 'predict':
        from inference.predict import predict_single_image
        predict_single_image(config)

if __name__ == '__main__':
    main()