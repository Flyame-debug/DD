import logging
import sys
import os
from pathlib import Path
from datetime import datetime

class ColabFormatter(logging.Formatter):
    """专为Colab环境优化的日志格式器，使用颜色和简洁格式"""
    
    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[0;36m',    # 青色
        'INFO': '\033[0;32m',     # 绿色
        'WARNING': '\033[1;33m',  # 黄色
        'ERROR': '\033[1;31m',    # 红色
        'CRITICAL': '\033[1;41m', # 白字红底
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record):
        # 添加颜色
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
            record.msg = f"{color}{record.msg}{reset}"
        
        # 简化长路径，只显示文件名
        if hasattr(record, 'pathname') and record.pathname:
            record.pathname = os.path.basename(record.pathname)
        
        return super().format(record)

def setup_logger(name, log_dir=None, level=logging.INFO, console=True, file=True):
    """
    设置并返回一个配置好的logger
    
    Args:
        name: logger名称
        log_dir: 日志文件目录，如果为None则不保存到文件
        level: 日志级别 (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: 是否输出到控制台
        file: 是否保存到文件
    
    Returns:
        logging.Logger: 配置好的logger实例
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建格式器
    console_formatter = ColabFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 文件handler
    if file and log_dir:
        # 确保日志目录存在
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件名（带时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'training_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件保存在: {log_file}")
    
    return logger

def log_epoch_results(logger, epoch, epoch_results, is_best=False):
    """
    记录一个epoch的训练结果
    
    Args:
        logger: logger实例
        epoch: 当前epoch数
        epoch_results: 包含训练结果的字典
        is_best: 是否是最佳模型
    """
    # 基本训练信息
    train_info = epoch_results.get('train', {})
    val_info = epoch_results.get('val', {})
    
    # 创建格式化的消息
    message_lines = []
    message_lines.append("=" * 70)
    message_lines.append(f"EPOCH {epoch:03d} {'⭐ BEST' if is_best else ''}")
    message_lines.append("-" * 70)
    
    # 训练损失
    if 'loss' in train_info:
        message_lines.append(f"训练损失: {train_info['loss']:.4f}")
    
    # 验证指标
    if 'loss' in val_info:
        message_lines.append(f"验证损失: {val_info['loss']:.4f}")
    
    if 'metrics' in val_info:
        metrics = val_info['metrics']
        message_lines.append(f"验证准确率: {metrics.get('accuracy', 0):.4f}")
        message_lines.append(f"验证F1分数: {metrics.get('f1', 0):.4f}")
        message_lines.append(f"验证AUC: {metrics.get('auc', 0):.4f}")
        message_lines.append(f"验证精确率: {metrics.get('precision', 0):.4f}")
        message_lines.append(f"验证召回率: {metrics.get('recall', 0):.4f}")
    
    # 学习率（如果有）
    if 'lr' in epoch_results:
        message_lines.append(f"学习率: {epoch_results['lr']:.6f}")
    
    # 训练时间（如果有）
    if 'time' in epoch_results:
        message_lines.append(f"训练时间: {epoch_results['time']:.1f}s")
    
    message_lines.append("=" * 70)
    
    # 记录日志
    for line in message_lines:
        logger.info(line)

def log_config(logger, config):
    """记录配置信息"""
    logger.info("=" * 60)
    logger.info("训练配置")
    logger.info("=" * 60)
    
    for section, values in config.items():
        if isinstance(values, dict):
            logger.info(f"[{section.upper()}]")
            for key, value in values.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info(f"{section}: {values}")
    
    logger.info("=" * 60)

def log_model_summary(logger, model):
    """记录模型概要信息"""
    logger.info("=" * 60)
    logger.info("模型架构")
    logger.info("=" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型: {model.__class__.__name__}")
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    logger.info(f"冻结参数量: {total_params - trainable_params:,}")
    
    logger.info("=" * 60)

def log_data_info(logger, dataloaders):
    """记录数据信息"""
    logger.info("=" * 60)
    logger.info("数据统计")
    logger.info("=" * 60)
    
    for name, loader in dataloaders.items():
        if loader is not None:
            dataset = loader.dataset
            logger.info(f"{name}:")
            logger.info(f"  样本数: {len(dataset):,}")
            logger.info(f"  Batch大小: {loader.batch_size}")
            logger.info(f"  Batch数: {len(loader)}")
            
            # 如果是训练集，记录类别分布（如果有的话）
            if name == 'train' and hasattr(dataset, 'get_class_distribution'):
                dist = dataset.get_class_distribution()
                logger.info(f"  类别分布: {dist}")
    
    logger.info("=" * 60)

# 单元测试
if __name__ == '__main__':
    print("测试 logger 模块...")
    
    # 测试1: 创建控制台logger
    console_logger = setup_logger('test_console', console=True, file=False)
    console_logger.debug("这是一条DEBUG消息")
    console_logger.info("这是一条INFO消息")
    console_logger.warning("这是一条WARNING消息")
    console_logger.error("这是一条ERROR消息")
    
    # 测试2: 创建带文件的logger
    test_log_dir = Path("./test_logs")
    file_logger = setup_logger('test_file', log_dir=test_log_dir, level=logging.DEBUG)
    file_logger.info("这条消息会同时输出到控制台和文件")
    
    # 测试3: 记录epoch结果
    epoch_results = {
        'train': {'loss': 0.1234},
        'val': {
            'loss': 0.0987,
            'metrics': {
                'accuracy': 0.8765,
                'f1': 0.8543,
                'auc': 0.9123,
                'precision': 0.8210,
                'recall': 0.7890
            }
        },
        'lr': 0.0001,
        'time': 45.6
    }
    
    log_epoch_results(file_logger, 1, epoch_results)
    log_epoch_results(file_logger, 2, epoch_results, is_best=True)
    
    # 测试4: 记录配置
    test_config = {
        'training': {
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001
        },
        'model': {
            'backbone': 'densenet121',
            'num_classes': 14
        }
    }
    
    log_config(file_logger, test_config)
    
    print("\n✅ logger.py 模块测试完成！")
    print(f"日志文件保存在: {test_log_dir.absolute()}")