import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(outputs, targets, threshold=0.5):
    """
    计算多标签分类的各项指标
    
    Args:
        outputs: 模型输出的logits或概率 (batch_size, num_classes)
        targets: 真实标签 (batch_size, num_classes)
        threshold: 将概率转换为二进制预测的阈值
    
    Returns:
        dict: 包含各项指标的字典
    """
    # 确保数据在CPU上并转换为numpy数组
    if torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    # 将logits转换为概率（如果使用BCEWithLogitsLoss）
    if outputs.min() < 0 or outputs.max() > 1:
        outputs = 1 / (1 + np.exp(-outputs))  # sigmoid转换
    
    # 应用阈值得到二进制预测
    predictions = (outputs > threshold).astype(int)
    
    metrics = {}
    
    # 计算准确率
    metrics['accuracy'] = accuracy_score(targets, predictions)
    
    # 计算精确率、召回率、F1分数（宏平均）
    metrics['precision'] = precision_score(targets, predictions, average='macro', zero_division=0)
    metrics['recall'] = recall_score(targets, predictions, average='macro', zero_division=0)
    metrics['f1'] = f1_score(targets, predictions, average='macro', zero_division=0)
    
    # 计算每个类别的AUC，然后取平均
    try:
        auc_scores = []
        for i in range(targets.shape[1]):
            if len(np.unique(targets[:, i])) > 1:  # 确保该类别有正负样本
                auc = roc_auc_score(targets[:, i], outputs[:, i])
                auc_scores.append(auc)
        metrics['auc'] = np.mean(auc_scores) if auc_scores else 0.0
    except ValueError:
        metrics['auc'] = 0.0
    
    # 计算每个样本的准确率（样本平均）
    sample_accuracy = np.mean(np.all(predictions == targets, axis=1))
    metrics['sample_accuracy'] = sample_accuracy
    
    # 计算每个类别的准确率（类别平均）
    class_accuracy = np.mean(np.all(predictions == targets, axis=0))
    metrics['class_accuracy'] = class_accuracy
    
    return metrics

def compute_confusion_matrix(predictions, targets):
    """
    计算多标签分类的混淆矩阵（按类别）
    
    Args:
        predictions: 预测标签 (batch_size, num_classes)
        targets: 真实标签 (batch_size, num_classes)
    
    Returns:
        dict: 每个类别的混淆矩阵字典
    """
    num_classes = targets.shape[1]
    confusion_matrices = {}
    
    for i in range(num_classes):
        pred = predictions[:, i]
        true = targets[:, i]
        
        tp = np.sum((pred == 1) & (true == 1))
        tn = np.sum((pred == 0) & (true == 0))
        fp = np.sum((pred == 1) & (true == 0))
        fn = np.sum((pred == 0) & (true == 1))
        
        confusion_matrices[f'class_{i}'] = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
    
    return confusion_matrices

def get_classification_report(outputs, targets, class_names=None, threshold=0.5):
    """
    生成详细的分类报告
    
    Args:
        outputs: 模型输出
        targets: 真实标签
        class_names: 类别名称列表
        threshold: 预测阈值
    
    Returns:
        str: 格式化的分类报告
    """
    if torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    if outputs.min() < 0 or outputs.max() > 1:
        outputs = 1 / (1 + np.exp(-outputs))
    
    predictions = (outputs > threshold).astype(int)
    num_classes = targets.shape[1]
    
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("多标签分类报告")
    report_lines.append("=" * 60)
    
    # 总体指标
    overall_metrics = calculate_metrics(outputs, targets, threshold)
    report_lines.append(f"\n总体指标:")
    report_lines.append(f"  准确率 (Accuracy): {overall_metrics['accuracy']:.4f}")
    report_lines.append(f"  精确率 (Precision): {overall_metrics['precision']:.4f}")
    report_lines.append(f"  召回率 (Recall): {overall_metrics['recall']:.4f}")
    report_lines.append(f"  F1分数 (F1-Score): {overall_metrics['f1']:.4f}")
    report_lines.append(f"  AUC: {overall_metrics['auc']:.4f}")
    report_lines.append(f"  样本准确率: {overall_metrics['sample_accuracy']:.4f}")
    report_lines.append(f"  类别准确率: {overall_metrics['class_accuracy']:.4f}")
    
    # 每个类别的指标
    report_lines.append(f"\n每个类别的详细指标:")
    report_lines.append("-" * 60)
    report_lines.append(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<10}")
    report_lines.append("-" * 60)
    
    for i in range(num_classes):
        pred = predictions[:, i]
        true = targets[:, i]
        
        if len(np.unique(true)) > 1:
            precision = precision_score(true, pred, zero_division=0)
            recall = recall_score(true, pred, zero_division=0)
            f1 = f1_score(true, pred, zero_division=0)
        else:
            precision = recall = f1 = 0.0
            
        support = np.sum(true)
        report_lines.append(f"{class_names[i]:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10.0f}")
    
    report_lines.append("=" * 60)
    
    return '\n'.join(report_lines)

# 单元测试
if __name__ == '__main__':
    # 测试数据
    batch_size, num_classes = 32, 14
    outputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # 测试计算指标
    metrics = calculate_metrics(outputs, targets)
    print("测试 calculate_metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 测试分类报告
    report = get_classification_report(outputs, targets)
    print(f"\n测试分类报告:\n{report}")
    
    print("\n✅ metrics.py 模块测试通过！")