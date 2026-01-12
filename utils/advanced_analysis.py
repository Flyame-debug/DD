# /content/DD/utils/advanced_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import torch
import cv2
from PIL import Image
import torch.nn.functional as F

class AdvancedAnalyzer:
    """高级分析工具类 - 用于生成PPT所需的深度分析"""
    
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.results = {}
        
    def analyze_class_imbalance(self, df, label_columns):
        """深入分析类别不平衡问题"""
        analysis = {}
        
        # 1. 计算基本统计
        total_samples = len(df)
        class_counts = df[label_columns].sum()
        class_percentages = (class_counts / total_samples * 100).round(2)
        
        # 2. 计算类别权重建议
        median_count = class_counts.median()
        suggested_weights = median_count / class_counts
        
        # 3. 分析疾病组合
        disease_combinations = []
        for idx, row in df.iterrows():
            diseases = [class_names[i] for i, val in enumerate(row[label_columns].values) if val == 1]
            if len(diseases) > 1:
                disease_combinations.append(tuple(sorted(diseases)))
        
        from collections import Counter
        combo_counts = Counter(disease_combinations)
        
        analysis['basic_stats'] = {
            'total_samples': total_samples,
            'class_counts': class_counts.to_dict(),
            'class_percentages': class_percentages.to_dict(),
            'imbalance_ratio': class_counts.max() / class_counts.min(),
            'median_count': median_count,
            'suggested_weights': suggested_weights.to_dict()
        }
        
        analysis['disease_combinations'] = {
            'total_combinations': len(combo_counts),
            'top_combinations': dict(combo_counts.most_common(10)),
            'single_disease_only': sum(df[label_columns].sum(axis=1) == 1),
            'multiple_diseases': sum(df[label_columns].sum(axis=1) > 1)
        }
        
        return analysis
    
    def plot_detailed_roc(self, y_true, y_pred, save_path=None):
        """绘制详细的ROC曲线，包含每个类别的表现"""
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()
        
        overall_fpr = []
        overall_tpr = []
        auc_scores = []
        
        for i, class_name in enumerate(self.class_names):
            if i >= len(axes):
                break
                
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            
            axes[i].plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC (AUC = {roc_auc:.3f})')
            axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'{class_name}\nAUC: {roc_auc:.3f}')
            axes[i].legend(loc="lower right")
            axes[i].grid(True, alpha=0.3)
            
            if i == 0:  # 收集第一个类别的数据作为整体曲线
                overall_fpr, overall_tpr = fpr, tpr
        
        # 隐藏多余的子图
        for i in range(len(self.class_names), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('每个类别的ROC曲线分析', fontsize=16, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"详细ROC曲线保存到: {save_path}")
        
        plt.show()
        
        # 返回AUC统计
        auc_stats = pd.DataFrame({
            'Class': self.class_names,
            'AUC': auc_scores,
            'Rank': pd.Series(auc_scores).rank(ascending=False).astype(int)
        }).sort_values('AUC', ascending=False)
        
        return auc_stats
    
    def analyze_failure_cases(self, test_loader, threshold=0.5):
        """深入分析模型失败案例"""
        self.model.eval()
        failure_cases = {
            'false_positives': {name: [] for name in self.class_names},
            'false_negatives': {name: [] for name in self.class_names},
            'hard_cases': []  # 难以分类的案例
        }
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                if isinstance(outputs, tuple):
                    predictions = torch.sigmoid(outputs[0])
                else:
                    predictions = torch.sigmoid(outputs)
                
                predictions_binary = (predictions > threshold).float()
                
                # 分析每个样本
                for i in range(len(images)):
                    pred = predictions_binary[i].cpu().numpy()
                    true = labels[i].cpu().numpy()
                    prob = predictions[i].cpu().numpy()
                    
                    # 检查每个类别的错误
                    for j, class_name in enumerate(self.class_names):
                        if pred[j] == 1 and true[j] == 0:  # 假阳性
                            failure_cases['false_positives'][class_name].append({
                                'batch_idx': batch_idx,
                                'sample_idx': i,
                                'confidence': prob[j],
                                'image': images[i].cpu()
                            })
                        elif pred[j] == 0 and true[j] == 1:  # 假阴性
                            failure_cases['false_negatives'][class_name].append({
                                'batch_idx': batch_idx,
                                'sample_idx': i,
                                'confidence': prob[j],
                                'image': images[i].cpu()
                            })
                    
                    # 识别难以分类的案例（预测概率接近阈值）
                    uncertainty = np.abs(prob - 0.5).mean()
                    if uncertainty < 0.1:  # 所有预测都接近0.5
                        failure_cases['hard_cases'].append({
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'probabilities': prob,
                            'true_labels': true,
                            'image': images[i].cpu()
                        })
        
        # 统计分析
        analysis = {
            'fp_counts': {k: len(v) for k, v in failure_cases['false_positives'].items()},
            'fn_counts': {k: len(v) for k, v in failure_cases['false_negatives'].items()},
            'hard_case_count': len(failure_cases['hard_cases']),
            'most_confusing_classes': sorted(
                [(k, len(failure_cases['false_positives'][k]) + len(failure_cases['false_negatives'][k])) 
                 for k in self.class_names],
                key=lambda x: x[1], reverse=True
            )[:5]
        }
        
        return failure_cases, analysis
    
    def visualize_attention(self, image, save_path=None):
        """可视化模型的注意力区域（使用Grad-CAM）"""
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        
        self.model.eval()
        
        # 获取目标层
        target_layer = None
        if hasattr(self.model, 'backbone'):
            # 对于EfficientNet，我们可以选择最后的卷积层
            target_layer = self.model.backbone._conv_head
        
        if target_layer is None:
            print("无法找到合适的层进行Grad-CAM可视化")
            return None
        
        # 创建Grad-CAM
        cam = GradCAM(model=self.model, target_layer=target_layer)
        
        # 准备输入
        input_tensor = image.unsqueeze(0).to(self.device)
        
        # 为每个类别生成热力图
        num_classes = len(self.class_names)
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()
        
        for class_idx in range(num_classes):
            if class_idx >= len(axes):
                break
            
            # 生成热力图
            grayscale_cam = cam(input_tensor=input_tensor, target_category=class_idx)
            
            # 可视化
            rgb_img = image.permute(1, 2, 0).cpu().numpy()
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
            
            axes[class_idx].imshow(visualization)
            axes[class_idx].set_title(f'{self.class_names[class_idx]}')
            axes[class_idx].axis('off')
        
        # 隐藏多余的子图
        for i in range(num_classes, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Grad-CAM 可视化: 模型关注的区域', fontsize=16, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"注意力可视化保存到: {save_path}")
        
        plt.show()
        
        return True
    
    def generate_comprehensive_report(self, test_results, save_path=None):
        """生成综合报告，用于PPT展示"""
        report = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {
                'name': self.model.__class__.__name__,
                'num_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'performance_summary': {
                'overall_auc': np.mean(test_results['auc_scores']),
                'overall_f1': np.mean(test_results['f1_scores']),
                'best_performing_class': self.class_names[np.argmax(test_results['auc_scores'])],
                'worst_performing_class': self.class_names[np.argmin(test_results['auc_scores'])],
                'performance_gap': np.max(test_results['auc_scores']) - np.min(test_results['auc_scores'])
            },
            'detailed_metrics': {
                'class_names': self.class_names,
                'auc_scores': test_results['auc_scores'],
                'f1_scores': test_results['f1_scores'],
                'precision_scores': test_results.get('precision_scores', []),
                'recall_scores': test_results.get('recall_scores', [])
            },
            'insights': {
                'structural_vs_textural': self._analyze_patterns(test_results),
                'recommendations': self._generate_recommendations(test_results)
            }
        }
        
        # 保存报告
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # 同时生成Markdown版本，方便复制到PPT
            md_path = save_path.replace('.json', '.md')
            self._save_markdown_report(report, md_path)
        
        return report
    
    def _analyze_patterns(self, test_results):
        """分析模型表现的模式"""
        # 根据医学知识分类
        structural_diseases = ['Pneumothorax', 'Cardiomegaly', 'Effusion', 'Mass']
        textural_diseases = ['Pneumonia', 'Infiltration', 'Consolidation', 'Edema']
        
        structural_auc = [test_results['auc_scores'][self.class_names.index(d)] 
                         for d in structural_diseases if d in self.class_names]
        textural_auc = [test_results['auc_scores'][self.class_names.index(d)] 
                       for d in textural_diseases if d in self.class_names]
        
        patterns = {
            'structural_diseases_avg_auc': np.mean(structural_auc) if structural_auc else 0,
            'textural_diseases_avg_auc': np.mean(textural_auc) if textural_auc else 0,
            'difference': np.mean(structural_auc) - np.mean(textural_auc) if structural_auc and textural_auc else 0,
            'interpretation': '模型对结构性病变识别更好' if np.mean(structural_auc) > np.mean(textural_auc) 
                            else '模型对纹理性病变识别更好'
        }
        
        return patterns
    
    def _generate_recommendations(self, test_results):
        """根据分析结果生成改进建议"""
        recommendations = []
        
        # 识别表现最差的类别
        worst_class_idx = np.argmin(test_results['auc_scores'])
        worst_class = self.class_names[worst_class_idx]
        worst_auc = test_results['auc_scores'][worst_class_idx]
        
        if worst_auc < 0.7:
            recommendations.append(
                f"**{worst_class}识别效果不佳(AUC={worst_auc:.3f})**：考虑数据增强、迁移学习或专门设计网络结构"
            )
        
        # 检查类别不平衡
        auc_std = np.std(test_results['auc_scores'])
        if auc_std > 0.1:
            recommendations.append(
                "**类别间性能差异大**：考虑使用更精细的类别权重调整或分层采样"
            )
        
        # 检查过拟合
        if hasattr(test_results, 'train_auc'):
            overfit_gap = test_results['train_auc'] - test_results['overall_auc']
            if overfit_gap > 0.15:
                recommendations.append(
                    f"**可能存在过拟合**(训练AUC-测试AUC={overfit_gap:.3f})：增加正则化、数据增强或早停"
                )
        
        recommendations.extend([
            "**工程优化**：部署时考虑模型量化、知识蒸馏以降低推理时间",
            "**临床验证**：需要与放射科医生合作进行真实世界验证",
            "**可解释性**：增加更多可视化工具帮助医生理解模型决策"
        ])
        
        return recommendations
    
    def _save_markdown_report(self, report, save_path):
        """保存Markdown格式的报告"""
        md_content = f"""# 模型分析报告

**生成时间**: {report['timestamp']}

## 1. 模型信息
- **模型名称**: {report['model_info']['name']}
- **总参数**: {report['model_info']['num_params']:,}
- **可训练参数**: {report['model_info']['trainable_params']:,}

## 2. 性能总结
- **平均AUC**: {report['performance_summary']['overall_auc']:.4f}
- **平均F1**: {report['performance_summary']['overall_f1']:.4f}
- **最佳类别**: {report['performance_summary']['best_performing_class']}
- **最差类别**: {report['performance_summary']['worst_performing_class']}
- **性能差距**: {report['performance_summary']['performance_gap']:.4f}

## 3. 详细指标

| 类别 | AUC | F1 | 精确率 | 召回率 |
|------|-----|----|--------|--------|
"""
        
        for i, class_name in enumerate(report['detailed_metrics']['class_names']):
            auc = report['detailed_metrics']['auc_scores'][i]
            f1 = report['detailed_metrics']['f1_scores'][i]
            precision = report['detailed_metrics']['precision_scores'][i] if i < len(report['detailed_metrics']['precision_scores']) else 0
            recall = report['detailed_metrics']['recall_scores'][i] if i < len(report['detailed_metrics']['recall_scores']) else 0
            
            md_content += f"| {class_name} | {auc:.4f} | {f1:.4f} | {precision:.4f} | {recall:.4f} |\n"
        
        md_content += f"""
## 4. 深度洞察

### 结构性 vs 纹理性病变
- **结构性病变平均AUC**: {report['insights']['structural_vs_textural']['structural_diseases_avg_auc']:.4f}
- **纹理性病变平均AUC**: {report['insights']['structural_vs_textural']['textural_diseases_avg_auc']:.4f}
- **差异**: {report['insights']['structural_vs_textural']['difference']:.4f}
- **解释**: {report['insights']['structural_vs_textural']['interpretation']}

## 5. 改进建议

"""
        
        for i, rec in enumerate(report['insights']['recommendations'], 1):
            md_content += f"{i}. {rec}\n"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"Markdown报告保存到: {save_path}")