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

# ========== 修复导入路径 ==========
# 添加项目根目录到Python路径
sys.path.insert(0, '/content/DD')

# 导入必要的模块
try:
    from models.model import MultiLabelModel, DenseNet121MultiLabel, EfficientNetB4MultiLabel
    print("✓ 成功导入模型模块")
except ImportError as e:
    print(f"导入模型模块失败: {e}")
    sys.exit(1)

try:
    from data.preprocess import load_and_preprocess_data
    print("✓ 成功导入数据预处理模块")
except ImportError as e:
    print(f"导入数据预处理模块失败: {e}")
    sys.exit(1)

try:
    from utils.logger import setup_logger
    print("✓ 成功导入日志模块")
except ImportError as e:
    print(f"导入日志模块失败: {e}")
    sys.exit(1)

def create_model(config, device):
    """创建模型 - 适配你的模型结构"""
    model_name = config['model']['backbone'].lower()
    num_classes = config['model']['num_classes']
    pretrained = config['model']['pretrained']
    dropout_rate = config['model']['dropout_rate']
    
    print(f"创建模型: {model_name}, 类别数: {num_classes}")
    
    # 根据模型名称选择不同的架构
    if model_name == 'densenet121':
        try:
            # 使用你自定义的DenseNet121MultiLabel
            model = DenseNet121MultiLabel(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout_rate=dropout_rate
            )
            print("✓ 使用DenseNet121MultiLabel")
        except Exception as e:
            print(f"创建DenseNet121失败: {e}")
            # 备用方案
            import torchvision.models as models
            model = models.densenet121(pretrained=pretrained)
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            print("✓ 使用torchvision DenseNet121")
            
    elif model_name == 'densenet50':
        # DenseNet50在torchvision中不存在，使用DenseNet121替代或通用MultiLabelModel
        try:
            # 使用通用MultiLabelModel，尝试densenet50
            model = MultiLabelModel(
                base_model='densenet50',
                num_classes=num_classes,
                pretrained=pretrained,
                dropout_rate=dropout_rate
            )
            print("✓ 使用MultiLabelModel with densenet50")
        except Exception as e:
            print(f"创建DenseNet50失败: {e}")
            # 使用DenseNet121作为替代
            import torchvision.models as models
            print("⚠ 使用DenseNet121替代DenseNet50")
            model = models.densenet121(pretrained=pretrained)
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        
    elif model_name.startswith('efficientnet'):
        # 提取版本号
        if '-' in model_name:
            version = model_name.split('-')[1]
        elif '_' in model_name:
            version = model_name.split('_')[1]
        else:
            version = 'b0'
        
        efficientnet_name = f'efficientnet-{version}'
        print(f"使用EfficientNet: {efficientnet_name}")
        
        try:
            # 使用通用MultiLabelModel
            model = MultiLabelModel(
                base_model=efficientnet_name,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout_rate=dropout_rate
            )
        except Exception as e:
            print(f"使用MultiLabelModel失败: {e}")
            # 备用方案
            from efficientnet_pytorch import EfficientNet
            if pretrained:
                model = EfficientNet.from_pretrained(efficientnet_name, num_classes=num_classes)
            else:
                model = EfficientNet.from_name(efficientnet_name, num_classes=num_classes)
            
            if dropout_rate > 0:
                in_features = model._fc.in_features
                model._fc = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    return model

def load_config(config_path):
    """加载配置 - 修复版本"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 检查是否有路径配置
    if 'paths' in config:
        # 如果已经有paths，直接使用
        pass
    elif 'paths_colab' in config and 'paths_local' in config:
        # 检查是否在Colab环境
        in_colab = 'COLAB_GPU' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ
        if in_colab:
            config['paths'] = config['paths_colab']
        else:
            config['paths'] = config['paths_local']
    elif 'paths_colab' in config:
        # 只有colab路径，直接使用
        config['paths'] = config['paths_colab']
    else:
        # 没有路径配置，创建默认
        config['paths'] = {
            'csv_path': './drive/MyDrive/filtered_labels.csv',
            'data_dir': './drive/MyDrive',
            'images_dir': './drive/MyDrive/images',
            'output_dir': './drive/MyDrive/outputs'
        }
    
    # 确保关键路径存在
    required_keys = ['csv_path', 'images_dir', 'output_dir']
    for key in required_keys:
        if key not in config['paths']:
            config['paths'][key] = f'./{key}'
    
    # 设置设备
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return config

def main():
    """主函数"""
    # 加载配置
    config_path = "/content/DD/config/config_optimized.yaml"
    if not os.path.exists(config_path):
        # 如果优化配置不存在，使用默认配置
        config_path = "/content/DD/config/config.yaml"
    
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device(config['device'])
    print(f"\n使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config['paths'].get('output_dir', f'/content/DD/outputs_{timestamp}')
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    config['paths']['output_dir'] = output_dir
    
    # 设置日志
    try:
        logger = setup_logger(output_dir)
    except:
        print("使用简单日志")
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    logger.info(f"开始训练，配置: {config}")
    
    try:
        # 1. 加载数据
        logger.info("加载数据...")
        train_df, val_df, test_df, label_columns, class_weights = load_and_preprocess_data(config)
        
        # 2. 创建模型
        logger.info("创建模型...")
        model = create_model(config, device)
        
        # 3. 检查是否需要创建数据集
        try:
            from data.dataset import ChestXRayDataset
            from data.dataset_cached import CachedChestXRayDataset
            
            # 使用缓存数据集
            logger.info("创建缓存数据集...")
            train_dataset = CachedChestXRayDataset(
                train_df,
                config['paths']['images_dir'],
                ChestXRayDataset.get_transforms(config, 'train'),
                'train',
                config['data']['image_size'][0],
                config['paths'].get('cache_dir'),
                use_cache=True
            )
            
            val_dataset = CachedChestXRayDataset(
                val_df,
                config['paths']['images_dir'],
                ChestXRayDataset.get_transforms(config, 'val'),
                'val',
                config['data']['image_size'][0],
                config['paths'].get('cache_dir'),
                use_cache=True
            )
            
        except ImportError as e:
            logger.warning(f"无法导入缓存数据集: {e}")
            logger.info("使用原始数据集...")
            from data.dataset import ChestXRayDataset
            
            train_dataset = ChestXRayDataset(
                train_df,
                config['paths']['images_dir'],
                ChestXRayDataset.get_transforms(config, 'train'),
                'train'
            )
            
            val_dataset = ChestXRayDataset(
                val_df,
                config['paths']['images_dir'],
                ChestXRayDataset.get_transforms(config, 'val'),
                'val'
            )
        
        # 4. 创建数据加载器
        batch_size = config['training']['batch_size']
        num_workers = config['training'].get('num_workers', 2)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"训练数据: {len(train_dataset)} 样本, {len(train_loader)} 批次")
        logger.info(f"验证数据: {len(val_dataset)} 样本, {len(val_loader)} 批次")
        
        # 5. 损失函数
        if isinstance(class_weights, (list, np.ndarray)):
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            logger.info(f"使用类别权重")
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # 6. 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 7. 检查是否有优化训练器
        try:
            from training.trainer_optimized import OptimizedTrainer
            logger.info("使用优化训练器...")
            trainer = OptimizedTrainer(config, device, logger)
            trainer.train(model, train_loader, val_loader, criterion, optimizer)
        except ImportError as e:
            logger.warning(f"无法导入优化训练器: {e}")
            logger.info("使用原始训练模块...")
            
            # 导入原始训练模块
            try:
                from training.train import train_epoch, validate_epoch
                
                best_val_loss = float('inf')
                patience_counter = 0
                early_stopping_patience = config['training']['early_stopping_patience']
                
                for epoch in range(config['training']['epochs']):
                    # 训练
                    train_loss, _ = train_epoch(
                        model, train_loader, criterion, optimizer, device, epoch
                    )
                    
                    # 验证
                    val_loss, val_metrics = validate_epoch(
                        model, val_loader, criterion, device, epoch
                    )
                    
                    logger.info(
                        f"Epoch {epoch+1}/{config['training']['epochs']} | "
                        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                    )
                    
                    # 早停
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"早停触发")
                            break
                            
            except ImportError as e:
                logger.error(f"无法导入原始训练函数: {e}")
                raise
        
        # 8. 保存最终模型
        final_model_path = os.path.join(output_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'label_columns': label_columns,
            'class_weights': class_weights
        }, final_model_path)
        
        logger.info(f"训练完成！模型保存到: {final_model_path}")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()