import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from pathlib import Path
import pandas as pd

# 设置中文字体和Seaborn样式（适配Colab环境）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

def plot_training_history(history, save_path=None, show=True):
    """
    绘制训练历史图表：损失和准确率变化
    
    Args:
        history: 包含训练历史的字典，应有'train_loss', 'val_loss', 'train_acc', 'val_acc'等键
        save_path: 图表保存路径，如果为None则不保存
        show: 是否显示图表
    """
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('训练历史可视化', fontsize=16, fontweight='bold')
    
    # 1. 损失曲线
    ax = axes[0, 0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('损失')
    ax.set_title('损失曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    ax = axes[0, 1]
    if 'train_acc' in history:
        ax.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    if 'val_acc' in history:
        ax.plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('准确率')
    ax.set_title('准确率曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. F1分数曲线
    ax = axes[0, 2]
    if 'train_f1' in history:
        ax.plot(epochs, history['train_f1'], 'b-', label='训练F1', linewidth=2, alpha=0.7)
    if 'val_f1' in history:
        ax.plot(epochs, history['val_f1'], 'r-', label='验证F1', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1分数')
    ax.set_title('F1分数曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. AUC曲线
    ax = axes[1, 0]
    if 'train_auc' in history:
        ax.plot(epochs, history['train_auc'], 'b-', label='训练AUC', linewidth=2, alpha=0.7)
    if 'val_auc' in history:
        ax.plot(epochs, history['val_auc'], 'r-', label='验证AUC', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('AUC曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 学习率变化（如果有）
    ax = axes[1, 1]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], 'g-', label='学习率', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('学习率')
        ax.set_title('学习率变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, '无学习率数据', ha='center', va='center', transform=ax.transAxes)
    
    # 6. 损失对比散点图
    ax = axes[1, 2]
    if 'train_loss' in history and 'val_loss' in history and len(history['train_loss']) > 0:
        ax.scatter(history['train_loss'], history['val_loss'], 
                  c=range(len(history['train_loss'])), cmap='viridis', alpha=0.6)
        ax.set_xlabel('训练损失')
        ax.set_ylabel('验证损失')
        ax.set_title('损失对比散点图')
        
        # 添加对角线
        min_val = min(min(history['train_loss']), min(history['val_loss']))
        max_val = max(max(history['train_loss']), max(history['val_loss']))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, '无足够数据', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_confusion_matrices(confusion_matrices, class_names=None, save_path=None, show=True):
    """
    绘制多标签分类的混淆矩阵热力图
    
    Args:
        confusion_matrices: 包含每个类别混淆矩阵的字典
        class_names: 类别名称列表
        save_path: 保存路径
        show: 是否显示
    """
    if not confusion_matrices:
        print("警告: 无混淆矩阵数据")
        return
    
    num_classes = len(confusion_matrices)
    if class_names is None:
        class_names = [f'类别{i}' for i in range(num_classes)]
    
    # 计算网格布局
    cols = min(4, num_classes)
    rows = (num_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    fig.suptitle('每个类别的混淆矩阵', fontsize=16, fontweight='bold')
    
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (class_key, cm_dict) in enumerate(confusion_matrices.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        class_idx = int(class_key.split('_')[-1]) if '_' in class_key else idx
        
        # 创建2x2混淆矩阵
        cm_data = np.array([
            [cm_dict.get('tn', 0), cm_dict.get('fp', 0)],
            [cm_dict.get('fn', 0), cm_dict.get('tp', 0)]
        ])
        
        # 绘制热力图
        im = ax.imshow(cm_data, cmap='Blues', interpolation='nearest')
        
        # 添加数值标签
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm_data[i, j]:,}', 
                       ha='center', va='center', 
                       color='black' if cm_data[i, j] < np.max(cm_data)/2 else 'white',
                       fontweight='bold')
        
        # 设置坐标轴
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['预测0', '预测1'])
        ax.set_yticklabels(['真实0', '真实1'])
        
        # 计算精确率和召回率
        tp, fp, fn = cm_dict.get('tp', 0), cm_dict.get('fp', 0), cm_dict.get('fn', 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        ax.set_title(f'{class_names[class_idx]}\n精确率: {precision:.3f}, 召回率: {recall:.3f}')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for idx in range(num_classes, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_roc_curves(fpr_dict, tpr_dict, auc_dict, class_names=None, save_path=None, show=True):
    """
    绘制ROC曲线
    
    Args:
        fpr_dict: 每个类别的假阳性率字典
        tpr_dict: 每个类别的真阳性率字典
        auc_dict: 每个类别的AUC值字典
        class_names: 类别名称
        save_path: 保存路径
        show: 是否显示
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制每个类别的ROC曲线
    for class_idx, (class_key, fpr) in enumerate(fpr_dict.items()):
        tpr = tpr_dict.get(class_key, [])
        auc = auc_dict.get(class_key, 0)
        
        if len(fpr) > 0 and len(tpr) > 0:
            if class_names and class_idx < len(class_names):
                label = f'{class_names[class_idx]} (AUC = {auc:.3f})'
            else:
                label = f'类别{class_idx} (AUC = {auc:.3f})'
            
            plt.plot(fpr, tpr, linewidth=2, label=label)
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')
    
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('ROC曲线 - 多标签分类')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC曲线已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_class_distribution(labels, class_names=None, save_path=None, show=True):
    """
    绘制类别分布条形图
    
    Args:
        labels: 标签数据 (n_samples, n_classes)
        class_names: 类别名称
        save_path: 保存路径
        show: 是否显示
    """
    if len(labels.shape) != 2:
        print("警告: 标签数据格式不正确，应为 (n_samples, n_classes)")
        return
    
    num_classes = labels.shape[1]
    if class_names is None:
        class_names = [f'类别{i}' for i in range(num_classes)]
    
    # 计算每个类别的正样本数量
    positive_counts = np.sum(labels, axis=0)
    total_samples = labels.shape[0]
    positive_percentages = (positive_counts / total_samples) * 100
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左侧：数量条形图
    bars1 = ax1.bar(range(num_classes), positive_counts, color='skyblue')
    ax1.set_xlabel('类别')
    ax1.set_ylabel('正样本数量')
    ax1.set_title('每个类别的正样本数量')
    ax1.set_xticks(range(num_classes))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    
    # 在条形上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 右侧：百分比条形图
    bars2 = ax2.bar(range(num_classes), positive_percentages, color='lightcoral')
    ax2.set_xlabel('类别')
    ax2.set_ylabel('正样本百分比 (%)')
    ax2.set_title('每个类别的正样本百分比')
    ax2.set_xticks(range(num_classes))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    
    # 在条形上添加百分比标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 添加总体统计信息
    fig.suptitle(f'类别分布统计 (总样本数: {total_samples:,})', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"类别分布图已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # 返回统计数据
    distribution_stats = {
        'total_samples': total_samples,
        'positive_counts': positive_counts.tolist(),
        'positive_percentages': positive_percentages.tolist(),
        'max_class': class_names[np.argmax(positive_counts)] if class_names else np.argmax(positive_counts),
        'min_class': class_names[np.argmin(positive_counts)] if class_names else np.argmin(positive_counts),
        'imbalance_ratio': np.max(positive_counts) / np.min(positive_counts) if np.min(positive_counts) > 0 else float('inf')
    }
    
    return distribution_stats

def plot_prediction_samples(images, predictions, targets, class_names=None, 
                           num_samples=10, save_path=None, show=True):
    """
    绘制预测样本对比图
    
    Args:
        images: 图像数据
        predictions: 预测结果
        targets: 真实标签
        class_names: 类别名称
        num_samples: 显示的样本数量
        save_path: 保存路径
        show: 是否显示
    """
    if len(images) == 0:
        print("警告: 无图像数据")
        return
    
    num_samples = min(num_samples, len(images))
    
    # 计算网格布局
    rows = int(np.ceil(num_samples / 5))
    cols = min(5, num_samples)
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle('预测样本对比', fontsize=16, fontweight='bold')
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # 显示图像
        if len(images[idx].shape) == 3 and images[idx].shape[0] in [1, 3]:
            # 调整通道顺序
            if images[idx].shape[0] == 3:
                img_to_show = np.transpose(images[idx], (1, 2, 0))
            else:
                img_to_show = images[idx][0]
        else:
            img_to_show = images[idx]
        
        # 归一化到0-1范围
        img_min, img_max = img_to_show.min(), img_to_show.max()
        if img_max > img_min:
            img_to_show = (img_to_show - img_min) / (img_max - img_min)
        
        ax.imshow(img_to_show, cmap='gray' if len(img_to_show.shape) == 2 else None)
        ax.axis('off')
        
        # 获取预测和真实标签
        pred = predictions[idx]
        target = targets[idx]
        
        # 计算正确预测的类别
        if len(pred.shape) == 1 and len(target.shape) == 1:
            correct = np.sum((pred > 0.5) == (target > 0.5))
            total = len(pred)
            
            # 显示准确率
            ax.set_title(f'样本 {idx}\n正确: {correct}/{total} ({correct/total*100:.1f}%)', 
                        fontsize=10)
            
            # 用颜色标记预测状态
            accuracy = correct / total
            if accuracy == 1.0:
                ax.spines['bottom'].set_color('green')
                ax.spines['top'].set_color('green')
                ax.spines['left'].set_color('green')
                ax.spines['right'].set_color('green')
                ax.spines['bottom'].set_linewidth(3)
                ax.spines['top'].set_linewidth(3)
                ax.spines['left'].set_linewidth(3)
                ax.spines['right'].set_linewidth(3)
            elif accuracy >= 0.7:
                ax.spines['bottom'].set_color('orange')
                ax.spines['top'].set_color('orange')
                ax.spines['left'].set_color('orange')
                ax.spines['right'].set_color('orange')
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['top'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)
                ax.spines['right'].set_linewidth(2)
            else:
                ax.spines['bottom'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['left'].set_color('red')
                ax.spines['right'].set_color('red')
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['top'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)
                ax.spines['right'].set_linewidth(2)
    
    # 隐藏多余的子图
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"预测样本图已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def save_all_visualizations(history, output_dir, config=None):
    """
    保存所有可视化图表
    
    Args:
        history: 训练历史
        output_dir: 输出目录
        config: 配置信息
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"保存可视化图表到: {output_dir}")
    
    # 1. 训练历史图
    history_path = output_dir / 'training_history.png'
    plot_training_history(history, save_path=history_path, show=False)
    
    # 2. 如果历史中有混淆矩阵数据
    if 'confusion_matrices' in history:
        cm_path = output_dir / 'confusion_matrices.png'
        plot_confusion_matrices(
            history['confusion_matrices'], 
            save_path=cm_path, 
            show=False
        )
    
    # 3. 如果历史中有ROC数据
    if 'fpr' in history and 'tpr' in history and 'auc' in history:
        roc_path = output_dir / 'roc_curves.png'
        plot_roc_curves(
            history['fpr'],
            history['tpr'],
            history['auc'],
            save_path=roc_path,
            show=False
        )
    
    # 4. 如果配置中有类别信息，创建类别分布图
    if config and 'data' in config and 'class_names' in config['data']:
        # 这里需要实际的标签数据，你可能需要从数据集中获取
        pass
    
    print(f"所有可视化图表已保存到: {output_dir}")

# 单元测试
if __name__ == '__main__':
    print("测试 visualization 模块...")
    
    # 创建测试数据
    epochs = 20
    test_history = {
        'train_loss': np.random.rand(epochs) * 0.5 + 0.3,
        'val_loss': np.random.rand(epochs) * 0.3 + 0.2,
        'train_acc': np.random.rand(epochs) * 0.2 + 0.7,
        'val_acc': np.random.rand(epochs) * 0.1 + 0.75,
        'train_f1': np.random.rand(epochs) * 0.15 + 0.65,
        'val_f1': np.random.rand(epochs) * 0.1 + 0.7,
        'train_auc': np.random.rand(epochs) * 0.1 + 0.8,
        'val_auc': np.random.rand(epochs) * 0.05 + 0.85,
        'lr': np.linspace(0.001, 0.0001, epochs)
    }
    
    # 测试训练历史图
    print("1. 测试训练历史图...")
    plot_training_history(test_history, show=False)
    
    # 测试混淆矩阵
    print("2. 测试混淆矩阵图...")
    test_cm = {
        'class_0': {'tp': 45, 'tn': 200, 'fp': 5, 'fn': 10},
        'class_1': {'tp': 30, 'tn': 180, 'fp': 20, 'fn': 15},
        'class_2': {'tp': 50, 'tn': 190, 'fp': 10, 'fn': 5}
    }
    plot_confusion_matrices(test_cm, show=False)
    
    # 测试类别分布图
    print("3. 测试类别分布图...")
    test_labels = np.random.randint(0, 2, (1000, 5))
    stats = plot_class_distribution(test_labels, show=False)
    print(f"分布统计: {stats}")
    
    print("\n✅ visualization.py 模块测试完成！")