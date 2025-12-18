import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data.dataset import ChestXRayDataset
from data.preprocess import load_and_preprocess_data
from models.model import DenseNet121MultiLabel, EfficientNetB4MultiLabel
from training.losses import WeightedBCELoss, FocalLoss
from training.metrics import calculate_metrics
from utils.logger import setup_logger
from utils.visualization import plot_training_history

def setup_device():
    """设置训练设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    return device

def create_model(config, device):
    """创建模型"""
    model_name = config['model']['backbone'].lower()
    num_classes = config['model']['num_classes']
    pretrained = config['model']['pretrained']
    dropout_rate = config['model']['dropout_rate']
    
    if model_name == 'densenet121':
        base_model = DenseNet121MultiLabel(
            num_classes=num_classes, 
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif model_name == 'efficientnet-b4':
        base_model = EfficientNetB4MultiLabel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    model = MultiLabelModel(base_model, num_classes=num_classes)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 记录
        running_loss += loss.item()
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})
        
        # 记录到TensorBoard
        if writer and batch_idx % 50 == 0:
            writer.add_scalar('Train/batch_loss', loss.item(), 
                             epoch * len(dataloader) + batch_idx)
    
    epoch_loss = running_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 计算训练指标
    metrics = calculate_metrics(all_labels, all_preds, threshold=0.5)
    
    return epoch_loss, metrics

def validate_epoch(model, dataloader, criterion, device, epoch, writer=None):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 计算验证指标
    metrics = calculate_metrics(all_labels, all_preds, threshold=0.5)
    
    # 记录到TensorBoard
    if writer:
        writer.add_scalar('Val/loss', epoch_loss, epoch)
        writer.add_scalar('Val/auc_mean', metrics['auc_mean'], epoch)
        writer.add_scalar('Val/f1_mean', metrics['f1_mean'], epoch)
    
    return epoch_loss, metrics

def train_model(config):
    """主训练函数"""
    
    # 设置设备
    device = setup_device()
    
    # 设置日志
    logger = setup_logger(config['paths']['output_dir'])
    logger.info(f"开始训练，配置文件: {config}")
    
    # 创建TensorBoard记录器
    writer = SummaryWriter(os.path.join(config['paths']['output_dir'], 'tensorboard'))
    
    # 加载和预处理数据
    logger.info("加载数据...")
    train_df, val_df, test_df, class_weights = load_and_preprocess_data(config)
    
    # 创建数据集
    train_dataset = ChestXRayDataset(
        train_df, 
        config['paths']['images_dir'],
        transform=ChestXRayDataset.get_transforms(config, 'train'),
        phase='train'
    )
    
    val_dataset = ChestXRayDataset(
        val_df,
        config['paths']['images_dir'],
        transform=ChestXRayDataset.get_transforms(config, 'val'),
        phase='val'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    logger.info("创建模型...")
    model = create_model(config, device)
    
    # 定义损失函数
    if config.get('use_focal_loss', False):
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        # 使用加权BCE损失
        weights = torch.stack([class_weights[i][1] for i in range(len(class_weights))]).to(device)
        criterion = WeightedBCELoss(pos_weight=weights)
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    logger.info("开始训练循环...")
    best_val_auc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'train_f1': [], 'val_f1': []
    }
    
    for epoch in range(config['training']['epochs']):
        start_time = time.time()
        
        # 训练
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # 验证
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # 学习率调度
        scheduler.step(val_metrics['auc_mean'])
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_metrics['auc_mean'])
        history['val_auc'].append(val_metrics['auc_mean'])
        history['train_f1'].append(train_metrics['f1_mean'])
        history['val_f1'].append(val_metrics['f1_mean'])
        
        # 打印进度
        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{config['training']['epochs']} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train AUC: {train_metrics['auc_mean']:.4f} | Val AUC: {val_metrics['auc_mean']:.4f} | "
            f"Train F1: {train_metrics['f1_mean']:.4f} | Val F1: {val_metrics['f1_mean']:.4f}"
        )
        
        # 保存最佳模型
        if val_metrics['auc_mean'] > best_val_auc:
            best_val_auc = val_metrics['auc_mean']
            patience_counter = 0
            
            # 保存模型
            model_path = os.path.join(config['paths']['output_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'val_f1': val_metrics['f1_mean'],
                'config': config,
                'class_names': train_dataset.class_names,
            }, model_path)
            logger.info(f"保存最佳模型到: {model_path}, AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
        
        # 定期保存检查点
        if (epoch + 1) % config['training']['save_checkpoint_freq'] == 0:
            checkpoint_path = os.path.join(
                config['paths']['output_dir'], 
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['auc_mean'],
                'config': config,
            }, checkpoint_path)
        
        # 早停
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"早停触发，最佳AUC: {best_val_auc:.4f}")
            break
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 可视化训练历史
    plot_training_history(history, config['paths']['output_dir'])
    
    logger.info("训练完成！")
    
    # 返回最佳模型路径
    return os.path.join(config['paths']['output_dir'], 'best_model.pth')