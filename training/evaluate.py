import torch
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from data.dataset import ChestXRayDataset
from models.model import create_model
from training.metrics import calculate_metrics
from utils.visualization import plot_roc_curves, plot_confusion_matrices

def evaluate_model(config, model_path=None):
    """评估模型性能"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 如果没有提供模型路径，使用最佳模型
    if model_path is None:
        model_path = os.path.join(config['paths']['output_dir'], 'best_model.pth')
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型
    model = create_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试数据
    test_df = pd.read_csv(os.path.join(config['paths']['output_dir'], 'test_split.csv'))
    
    test_dataset = ChestXRayDataset(
        test_df,
        config['paths']['images_dir'],
        transform=ChestXRayDataset.get_transforms(config, 'val'),
        phase='test'
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # 评估
    print("评估模型...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds, threshold=0.5)
    
    print("\n" + "="*50)
    print("模型评估结果")
    print("="*50)
    print(f"平均AUC: {metrics['auc_mean']:.4f}")
    print(f"平均F1: {metrics['f1_mean']:.4f}")
    
    # 打印每个类别的指标
    print("\n每个类别的性能:")
    class_names = test_dataset.class_names
    for i, class_name in enumerate(class_names):
        print(f"{class_name:20s} AUC: {metrics['auc_per_class'][i]:.4f}  "
              f"F1: {metrics['f1_per_class'][i]:.4f}")
    
    # 可视化
    output_dir = config['paths']['output_dir']
    
    # 绘制ROC曲线
    plot_roc_curves(
        all_labels, all_preds, class_names,
        save_path=os.path.join(output_dir, 'roc_curves.png')
    )
    
    # 绘制混淆矩阵
    plot_confusion_matrices(
        all_labels, all_preds, class_names,
        threshold=0.5,
        save_path=os.path.join(output_dir, 'confusion_matrices.png')
    )
    
    # 保存评估结果
    results_df = pd.DataFrame({
        'Class': class_names,
        'AUC': metrics['auc_per_class'],
        'F1': metrics['f1_per_class']
    })
    results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    
    return metrics