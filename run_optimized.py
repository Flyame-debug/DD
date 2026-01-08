#!/usr/bin/env python
# /content/DD/run_optimized.py
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset_cached import CachedChestXRayDataset
from data.preprocess import load_and_preprocess_data
from models.model import create_model
from training.trainer_optimized import OptimizedTrainer
from utils.logger import setup_logger

def load_config(config_path):
    """加载配置"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 合并路径
    if 'paths_colab' in config and 'CUDA_VISIBLE_DEVICES' in os.environ:
        config['paths'] = config['paths_colab']
    else:
        config['paths'] = config['paths_local']
    
    # 设置设备
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return config

def main():
    """主函数"""
    # 加载配置
    config_path = "/content/DD/config/config_optimized.yaml"
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device(config['device'])
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['paths']['output_dir'] = os.path.join(config['paths']['output_dir'], timestamp)
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    logger = setup_logger(config['paths']['output_dir'])
    logger.info(f"开始优化训练，配置: {config}")
    
    try:
        # 1. 加载数据（快速版）
        logger.info("快速加载数据...")
        train_df, val_df, test_df, label_columns, class_weights = load_and_preprocess_data(config)
        
        # 2. 创建缓存数据集
        logger.info("创建缓存数据集...")
        train_dataset = CachedChestXRayDataset(
            train_df,
            config['paths']['images_dir'],
            CachedChestXRayDataset.get_transforms(config, 'train'),
            'train',
            config['data']['image_size'][0],
            config['paths'].get('cache_dir'),
            use_cache=True
        )
        
        val_dataset = CachedChestXRayDataset(
            val_df,
            config['paths']['images_dir'],
            CachedChestXRayDataset.get_transforms(config, 'val'),
            'val',
            config['data']['image_size'][0],
            config['paths'].get('cache_dir'),
            use_cache=True
        )
        
        # 3. 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory'],
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
        
        logger.info(f"训练数据: {len(train_dataset)} 样本, {len(train_loader)} 批次")
        logger.info(f"验证数据: {len(val_dataset)} 样本, {len(val_loader)} 批次")
        
        # 4. 创建模型
        logger.info("创建模型...")
        model = create_model(config, device)
        
        # 5. 损失函数和优化器
        if isinstance(class_weights, (list, np.ndarray)):
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 6. 训练
        trainer = OptimizedTrainer(config, device, logger)
        trainer.train(model, train_loader, val_loader, criterion, optimizer)
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()