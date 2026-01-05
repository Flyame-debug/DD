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
from PIL import Image
import glob

from data.dataset import ChestXRayDataset
from data.preprocess import load_and_preprocess_data
from models.model import DenseNet121MultiLabel, EfficientNetB4MultiLabel
from training.losses import WeightedBCELoss, FocalLoss
from training.metrics import calculate_metrics
from utils.logger import setup_logger
from utils.visualization import plot_training_history

class MultiLabelModel(nn.Module):
    """多标签分类模型包装器"""
    def __init__(self, base_model, num_classes):
        super(MultiLabelModel, self).__init__()
        self.base_model = base_model
        
    def forward(self, x):
        return self.base_model(x)

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
    elif model_name.startswith('efficientnet'):
        # 支持所有efficientnet变体
        from efficientnet_pytorch import EfficientNet
        
        # 提取版本号，如 'b0', 'b1', 'b4' 等
        if '-' in model_name:
            # 格式: efficientnet-b4
            version = model_name.split('-')[1]
        elif '_' in model_name:
            # 格式: efficientnet_b0
            version = model_name.split('_')[1]
        else:
            # 默认使用b0
            version = 'b0'
        
        # 构建完整的模型名称
        efficientnet_name = f'efficientnet-{version}'
        
        if pretrained:
            base_model = EfficientNet.from_pretrained(efficientnet_name, num_classes=num_classes)
        else:
            base_model = EfficientNet.from_name(efficientnet_name, num_classes=num_classes)
        
        # 修改分类器以添加dropout
        if dropout_rate > 0:
            in_features = base_model._fc.in_features
            base_model._fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes)
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
        all_preds.append(outputs.detach())
        all_labels.append(labels.detach())
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})
        
        # 记录到TensorBoard
        if writer and batch_idx % 50 == 0:
            writer.add_scalar('Train/batch_loss', loss.item(), 
                             epoch * len(dataloader) + batch_idx)
    
    epoch_loss = running_loss / len(dataloader)
    
    # 计算训练指标 - 修复参数顺序
    if all_preds and all_labels:
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        # 注意：第一个参数是预测，第二个参数是标签
        metrics = calculate_metrics(all_preds, all_labels, threshold=0.5)
        
        # 确保键名兼容
        if 'auc_mean' not in metrics and 'auc' in metrics:
            metrics['auc_mean'] = metrics['auc']
        if 'f1_mean' not in metrics and 'f1' in metrics:
            metrics['f1_mean'] = metrics['f1']
    else:
        metrics = {
            'auc_mean': 0.0,
            'f1_mean': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
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
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
            
            pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    
    # 计算验证指标 - 修复参数顺序
    if all_preds and all_labels:
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        # 注意：第一个参数是预测，第二个参数是标签
        metrics = calculate_metrics(all_preds, all_labels, threshold=0.5)
        
        # 确保键名兼容
        if 'auc_mean' not in metrics and 'auc' in metrics:
            metrics['auc_mean'] = metrics['auc']
        if 'f1_mean' not in metrics and 'f1' in metrics:
            metrics['f1_mean'] = metrics['f1']
    else:
        metrics = {
            'auc_mean': 0.0,
            'f1_mean': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    # 记录到TensorBoard
    if writer:
        writer.add_scalar('Val/loss', epoch_loss, epoch)
        writer.add_scalar('Val/auc_mean', metrics.get('auc_mean', metrics.get('auc', 0.0)), epoch)
        writer.add_scalar('Val/f1_mean', metrics.get('f1_mean', metrics.get('f1', 0.0)), epoch)
    
    return epoch_loss, metrics

def test_model(model, test_loader, device, config):
    """测试模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > 0.5).float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 计算测试指标
    test_metrics = calculate_metrics(all_labels, all_probs, threshold=0.5)
    
    return test_metrics

def is_valid_image(file_path):
    """检查图片是否有效"""
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证文件完整性
            return True
    except:
        return False

def find_image_path(images_dir, image_name):
    """查找图片路径，支持递归查找"""
    # 首先尝试直接路径
    direct_path = os.path.join(images_dir, image_name)
    if os.path.exists(direct_path) and is_valid_image(direct_path):
        return direct_path
    
    # 尝试查找不同的文件扩展名
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        if not image_name.lower().endswith(ext.lower()):
            test_path = os.path.join(images_dir, image_name + ext)
            if os.path.exists(test_path) and is_valid_image(test_path):
                return test_path
    
    # 递归查找
    pattern = os.path.join(images_dir, '**', image_name)
    found_files = glob.glob(pattern, recursive=True)
    for file_path in found_files:
        if is_valid_image(file_path):
            return file_path
    
    # 尝试递归查找带扩展名
    for ext in ['.png', '.jpg', '.jpeg']:
        pattern = os.path.join(images_dir, '**', image_name + ext)
        found_files = glob.glob(pattern, recursive=True)
        for file_path in found_files:
            if is_valid_image(file_path):
                return file_path
    
    return None

def create_dataset_with_path_fix(df, images_dir, transform, phase, logger, config, label_columns):
    """创建数据集，自动修复图片路径"""
    # 收集有效的样本
    valid_indices = []
    image_paths = []
    
    logger.info(f"开始查找和验证图片文件，共 {len(df)} 个样本...")
    
    image_size = config['data']['image_size'][0]  # 获取图像尺寸
    
    # 重置DataFrame索引，确保索引是连续的
    df = df.reset_index(drop=True)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"查找{phase}图片"):
        image_name = row['Image Index']
        img_path = find_image_path(images_dir, image_name)
        
        if img_path:
            valid_indices.append(idx)
            image_paths.append(img_path)
        else:
            # 尝试其他可能的文件名格式
            # 移除可能的额外扩展名
            base_name = os.path.splitext(image_name)[0]
            img_path = find_image_path(images_dir, base_name)
            if img_path:
                valid_indices.append(idx)
                image_paths.append(img_path)
    
    logger.info(f"找到 {len(valid_indices)} 个有效样本（{len(df) - len(valid_indices)} 个缺失/损坏）")
    
    if len(valid_indices) == 0:
        raise ValueError("没有找到任何有效的图片文件！")
    
    # 使用有效的索引提取DataFrame行
    # 确保索引在范围内
    valid_indices = [idx for idx in valid_indices if idx < len(df)]
    
    # 创建有效的DataFrame
    valid_df = df.iloc[valid_indices].copy()
    valid_df = valid_df.reset_index(drop=True)
    
    # 只保留必要的列：图像索引、路径和标签列
    columns_to_keep = ['Image Index'] + list(label_columns)
    if 'image_path' in valid_df.columns:
        columns_to_keep.insert(1, 'image_path')
    valid_df = valid_df[columns_to_keep]
    
    logger.info(f"DataFrame形状: {valid_df.shape}")
    logger.info(f"DataFrame列: {valid_df.columns.tolist()}")
    
    # 创建自定义数据集类
    class FixedChestXRayDataset(ChestXRayDataset):
        def __init__(self, df, image_paths, transform=None, phase='train', image_size=512):
            # 保存df到实例变量
            self.df = df
            self.image_paths = image_paths
            self.image_size = image_size
            self.invalid_image_cache = {}  # 缓存无效图片，避免重复打开
            self.phase = phase
            
            # 调用父类初始化，但传递空字符串作为图像目录
            # 因为我们使用自己的image_paths
            super().__init__(df, '', transform, phase)
            
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            
            # 检查缓存中是否有无效图片记录
            if img_path in self.invalid_image_cache:
                image = self.invalid_image_cache[img_path]
            else:
                try:
                    image = Image.open(img_path).convert('RGB')
                    # 验证图片是否可以正常读取
                    image.load()  # 确保图片完全加载
                except Exception as e:
                    # 创建黑色图像作为替代
                    print(f"无法读取图片 {img_path}: {e}，使用黑色图像替代")
                    image = Image.new('RGB', (self.image_size, self.image_size), color=0)
                    self.invalid_image_cache[img_path] = image
            
            # 获取标签 - 跳过前几列（'Image Index'和可能的'image_path'）
            # 找到第一个标签列的索引
            label_start_idx = 1  # 跳过'Image Index'
            if 'image_path' in self.df.columns:
                label_start_idx = 2  # 跳过'Image Index'和'image_path'
            
            labels = self.df.iloc[idx, label_start_idx:].values.astype(np.float32)
            
            if self.transform:
                # 检查transform是否是albumentations类型
                # 如果是，需要以字典形式传递
                try:
                    # 尝试直接调用，如果是torchvision的transform
                    image = self.transform(image)
                except (KeyError, TypeError):
                    # 如果是albumentations的transform，需要转换为numpy数组并以字典形式传递
                    import albumentations as A
                    if isinstance(self.transform, A.Compose):
                        # 转换为numpy数组
                        image_np = np.array(image)
                        # 以字典形式传递
                        transformed = self.transform(image=image_np)
                        image = transformed['image']
                    else:
                        # 其他情况，尝试直接调用
                        image = self.transform(image)
                
            return image, labels
        
        def __len__(self):
            return len(self.df)
    
    return FixedChestXRayDataset(valid_df, image_paths, transform, phase, image_size)

def train_model(config):
    """主训练函数"""
    
    # 设置设备
    device = setup_device()
    
    # 设置日志
    logger = setup_logger(config['paths']['output_dir'])
    logger.info(f"开始训练，配置文件: {config}")
    
    # 创建输出目录
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    # 创建TensorBoard记录器
    writer = SummaryWriter(os.path.join(config['paths']['output_dir'], 'tensorboard'))
    
    # 加载和预处理数据
    logger.info("加载数据...")
    train_df, val_df, test_df, label_columns, class_weights = load_and_preprocess_data(config)
    
    # 根据实际类别数更新配置
    actual_num_classes = len(label_columns)
    config['model']['num_classes'] = actual_num_classes
    logger.info(f"实际类别数: {actual_num_classes}")
    
    # 打印DataFrame结构以调试
    logger.info(f"训练DataFrame形状: {train_df.shape}")
    logger.info(f"训练DataFrame列: {train_df.columns.tolist()}")
    
    # 创建数据集 - 使用修复版本
    logger.info("创建训练数据集...")
    train_dataset = create_dataset_with_path_fix(
        train_df, 
        config['paths']['images_dir'],
        ChestXRayDataset.get_transforms(config, 'train'),
        'train',
        logger,
        config,
        label_columns
    )
    
    logger.info("创建验证数据集...")
    val_dataset = create_dataset_with_path_fix(
        val_df,
        config['paths']['images_dir'],
        ChestXRayDataset.get_transforms(config, 'val'),
        'val',
        logger,
        config,
        label_columns
    )
    
    # 检查数据集大小
    logger.info(f"训练数据集大小: {len(train_dataset)}")
    logger.info(f"验证数据集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,  # 减少worker数量以避免问题
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,  # 减少worker数量以避免问题
        pin_memory=True
    )
    
    # 检查数据加载器
    logger.info(f"训练数据加载器批次数量: {len(train_loader)}")
    logger.info(f"验证数据加载器批次数量: {len(val_loader)}")
    
    # 创建模型
    logger.info("创建模型...")
    model = create_model(config, device)
    
    # 定义损失函数
    if config.get('use_focal_loss', False):
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        # 使用加权BCE损失
        if isinstance(class_weights, np.ndarray) or isinstance(class_weights, list):
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            logger.info(f"使用类别权重: {[float(w) for w in class_weights]}")
            criterion = WeightedBCELoss(pos_weight=weights_tensor)
        else:
            logger.warning(f"class_weights 类型错误: {type(class_weights)}，使用默认损失函数")
            criterion = nn.BCEWithLogitsLoss()
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
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
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch+1}/{config['training']['epochs']} | "
            f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f} | "
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
                'class_names': label_columns,
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
            logger.info(f"保存检查点到: {checkpoint_path}")
        
        # 早停
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"早停触发，最佳AUC: {best_val_auc:.4f}")
            break
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 可视化训练历史
    plot_training_history(history, config['paths']['output_dir'])
    
    # 测试最佳模型
    logger.info("在测试集上评估最佳模型...")
    best_model_path = os.path.join(config['paths']['output_dir'], 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建测试数据集
        logger.info("创建测试数据集...")
        test_dataset = create_dataset_with_path_fix(
            test_df,
            config['paths']['images_dir'],
            ChestXRayDataset.get_transforms(config, 'val'),
            'test',
            logger,
            config,
            label_columns
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # 测试模型
        test_metrics = test_model(model, test_loader, device, config)
        logger.info(f"测试结果 - AUC: {test_metrics['auc_mean']:.4f}, F1: {test_metrics['f1_mean']:.4f}")
        
        # 保存测试结果
        test_results_path = os.path.join(config['paths']['output_dir'], 'test_results.txt')
        with open(test_results_path, 'w') as f:
            f.write(f"Test AUC: {test_metrics['auc_mean']:.4f}\n")
            f.write(f"Test F1: {test_metrics['f1_mean']:.4f}\n")
            f.write(f"Per-class AUC: {test_metrics['auc']}\n")
            f.write(f"Per-class F1: {test_metrics['f1']}\n")
    
    logger.info("训练完成！")
    
    # 返回最佳模型路径
    return os.path.join(config['paths']['output_dir'], 'best_model.pth')