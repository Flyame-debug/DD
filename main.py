import argparse
import os
import sys
import yaml
from pathlib import Path

def setup_environment(is_colab=False):
    """设置环境，导入必要的包"""
    if is_colab:
        print("Running in Google Colab environment")
        # 使用Python的subprocess模块来执行shell命令，这是正确的方法
        import subprocess
        import sys
        
        # 定义需要安装的包
        packages = [
            'torch',
            'torchvision', 
            'torchtext',
            'torchaudio',
            'efficientnet-pytorch',
            'albumentations'
        ]
        
        # 执行安装命令
        for package in packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {e}")

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
    
    # 设置环境（如果需要）
    setup_environment(args.colab)
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 根据环境选择路径
    if args.colab:
      config['paths'] = config['paths_colab']
      # Google Drive 现在应已在Colab环境手动挂载
      # 因此跳过脚本内的挂载尝试，仅检查路径是否存在
      expected_mount_point = config['paths']['drive_mount']  # 即 '/content/drive'
      if not os.path.exists(expected_mount_point):
        print(f"警告: Google Drive 似乎未在预期位置挂载 ({expected_mount_point})。")
        print("请确保已在Colab单元格中手动运行过: `from google.colab import drive; drive.mount('/content/drive')`")
      else:
        print(f"Google Drive 已挂载于: {expected_mount_point}")
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
    try:
        if args.mode == 'train':
            from training.train import train_model
            train_model(config)
        elif args.mode == 'test':
            from training.evaluate import evaluate_model
            evaluate_model(config)
        elif args.mode == 'predict':
            from inference.predict import predict_single_image
            predict_single_image(config)
    except ImportError as e:
        print(f"导入模块失败: {e}")
        print("请确保所有依赖包已正确安装，或检查模块路径是否正确。")
        sys.exit(1)
    except Exception as e:
        print(f"运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()