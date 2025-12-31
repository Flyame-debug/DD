import os
import sys
import torch
import gc
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """设置环境，清理内存，优化GPU使用"""
    print("=== 环境设置与内存优化 ===")
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 设置PyTorch内存优化环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"使用设备: {device}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # 检查GPU内存
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated() / 1e9
        free_memory = total_memory - allocated_memory
        
        logger.info(f"显存: {total_memory:.2f} GB")
        logger.info(f"已用显存: {allocated_memory:.2f} GB")
        logger.info(f"可用显存: {free_memory:.2f} GB")
        
        # 如果内存不足，发出警告
        if free_memory < 2.0:  # 少于2GB可用
            logger.warning(f"GPU内存不足！只有 {free_memory:.2f} GB 可用")
            return device, True  # 返回设备和内存不足标志
        else:
            return device, False
    else:
        device = torch.device('cpu')
        logger.info(f"使用设备: {device}")
        return device, False

def load_config():
    """加载配置，并根据GPU内存自动调整"""
    import json
    
    # 尝试加载JSON配置文件
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"从JSON文件加载配置: {config_path}")
        return config
    
    # 尝试加载YAML配置文件
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"从YAML文件加载配置: {config_path}")
        return config
    
    # 如果配置文件不存在，使用默认配置
    logger.warning("配置文件不存在，使用默认配置")
    return get_default_config()

def get_default_config():
    """获取默认配置"""
    return {
        "augmentation": {
            "train": {
                "random_brightness": 0.1,
                "random_contrast": 0.1,
                "random_horizontal_flip": True,
                "random_rotation": 10
            },
            "val": {
                "resize_only": True
            }
        },
        "data": {
            "image_size": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "test_split": 0.15,
            "train_split": 0.7,
            "val_split": 0.15
        },
        "model": {
            "backbone": "efficientnet_b0",
            "dropout_rate": 0.3,
            "num_classes": 14,
            "pretrained": True
        },
        "paths_colab": {
            "csv_path": "/content/drive/MyDrive/filtered_labels.csv",
            "data_dir": "/content/drive/MyDrive",
            "drive_mount": "/content/drive",
            "images_dir": "/content/drive/MyDrive/images",
            "output_dir": "/content/DD/outputs"
        },
        "paths_local": {
            "csv_path": "./data/chest_xray/labels.csv",
            "data_dir": "./data/chest_xray",
            "images_dir": "./data/chest_xray/images",
            "output_dir": "./outputs"
        },
        "training": {
            "batch_size": 8,
            "epochs": 30,
            "learning_rate": 0.0001,
            "weight_decay": 1e-05,
            "early_stopping_patience": 10,
            "save_checkpoint_freq": 5,
            "use_amp": True,
            "gradient_accumulation_steps": 2
        }
    }

def optimize_config_for_memory(config, memory_limited=False):
    """根据内存情况优化配置"""
    logger.info("根据内存情况优化配置...")
    
    if memory_limited:
        logger.warning("检测到GPU内存不足，使用低内存配置")
        config['data']['image_size'] = [128, 128]
        config['training']['batch_size'] = 4
        config['model']['backbone'] = 'resnet18'
        config['training']['use_amp'] = True
        config['training']['gradient_accumulation_steps'] = 4
    else:
        # 中等配置
        config['data']['image_size'] = [224, 224]
        config['training']['batch_size'] = 16
        config['model']['backbone'] = 'efficientnet_b0'
        config['training']['use_amp'] = True
    
    # 设置路径
    if 'COLAB_GPU' in os.environ:
        logger.info("检测到Colab环境，使用Colab路径配置")
        config['paths'] = config['paths_colab']
        os.makedirs(config['paths']['output_dir'], exist_ok=True)
    else:
        logger.info("检测到本地环境，使用本地路径配置")
        config['paths'] = config['paths_local']
        os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    logger.info(f"优化后图像尺寸: {config['data']['image_size']}")
    logger.info(f"优化后批次大小: {config['training']['batch_size']}")
    logger.info(f"优化后模型骨架: {config['model']['backbone']}")
    
    return config

def check_data_availability(config):
    """检查数据是否可用"""
    logger.info("检查数据可用性...")
    
    import glob
    
    # 检查CSV文件
    csv_path = config['paths']['csv_path']
    if not os.path.exists(csv_path):
        logger.error(f"CSV文件不存在: {csv_path}")
        return False
    
    # 检查图片目录
    images_dir = config['paths']['images_dir']
    if not os.path.exists(images_dir):
        logger.error(f"图片目录不存在: {images_dir}")
        return False
    
    # 检查是否有图片文件
    image_files = glob.glob(os.path.join(images_dir, '**', '*.png'), recursive=True)
    image_files.extend(glob.glob(os.path.join(images_dir, '**', '*.jpg'), recursive=True))
    image_files.extend(glob.glob(os.path.join(images_dir, '**', '*.jpeg'), recursive=True))
    
    if len(image_files) == 0:
        logger.error(f"在 {images_dir} 中未找到任何图片文件")
        return False
    
    logger.info(f"找到 {len(image_files)} 张图片")
    return True

def main():
    """主函数"""
    try:
        # 1. 设置环境
        device, memory_limited = setup_environment()
        
        # 2. 加载配置
        config = load_config()
        
        # 3. 根据内存优化配置
        config = optimize_config_for_memory(config, memory_limited)
        
        # 4. 检查数据
        if not check_data_availability(config):
            logger.error("数据不可用，程序退出")
            return
        
        # 5. 将设备信息添加到config中
        config['device'] = str(device)  # 保存为字符串，便于序列化
        
        # 6. 记录配置
        logger.info(f"开始训练，配置文件: {config}")
        
        # 7. 导入训练模块并开始训练
        from training.train import train_model
        logger.info("导入训练模块成功")
        
        # 8. 开始训练（只传递一个参数）
        train_model(config)
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"运行过程中发生错误: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()