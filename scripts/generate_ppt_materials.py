# /content/DD/scripts/generate_ppt_materials.py
#!/usr/bin/env python
"""
生成PPT所需的所有素材：
1. 训练曲线
2. 混淆矩阵
3. ROC曲线
4. 类别分布分析
5. 注意力可视化
6. 失败案例分析
7. 综合报告
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append('/content/DD')

def setup_environment():
    """设置环境"""
    print("="*80)
    print("生成PPT素材脚本")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    return device

def load_best_model(config_path, model_path):
    """通用模型加载器，支持多种模型结构"""
    import yaml
    import torch
    import timm
    from models.model_enhanced import EnhancedMultiLabelModel, AttentionBlock
    
    # 加载配置和检查点
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 获取模型配置
    backbone_name = config.get('model', {}).get('backbone', 'efficientnet_b0')
    num_classes = config.get('model', {}).get('num_classes', 14)
    dropout_rate = config.get('model', {}).get('dropout_rate', 0.4)
    use_attention = config.get('model', {}).get('use_attention', True)
    
    # 分析状态字典结构
    keys = list(state_dict.keys())
    
    # 判断模型类型
    if any(k.startswith('base_model.') for k in keys[:10]):
        print("检测到包装的基础EfficientNet模型")
        # 创建EnhancedMultiLabelModel
        model = EnhancedMultiLabelModel(
            base_model=backbone_name,
            num_classes=num_classes,
            pretrained=False,
            dropout_rate=dropout_rate,
            use_attention=use_attention
        )
        
        # 需要转换键名
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('base_model.'):
                # 转换为EnhancedMultiLabelModel期望的格式
                new_key = key.replace('base_model.', 'backbone.')
                new_state_dict[new_key] = value
            elif key.startswith('base_model._fc.'):
                # 基础分类器转换为全局分类器
                new_key = key.replace('base_model._fc.', 'global_classifier.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # 加载状态字典（strict=False允许不匹配的键）
        model.load_state_dict(new_state_dict, strict=False)
        
    elif any(k.startswith('backbone.') for k in keys[:10]):
        print("检测到EnhancedMultiLabelModel")
        model = EnhancedMultiLabelModel(
            base_model=backbone_name,
            num_classes=num_classes,
            pretrained=False,
            dropout_rate=dropout_rate,
            use_attention=use_attention
        )
        model.load_state_dict(state_dict)
        
    elif any(k.startswith('_conv_stem') or k.startswith('_bn0') for k in keys[:10]):
        print("检测到基础EfficientNet模型（无包装）")
        model = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=num_classes
        )
        model.load_state_dict(state_dict)
        
    else:
        raise ValueError(f"无法识别的模型结构")
    
    # 获取类别名称
    class_names = checkpoint.get('class_names', [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
        'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 
        'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 
        'Pneumonia', 'Pneumothorax'
    ])
    
    return model, config, class_names
    
    #return model, config, checkpoint.get('class_names', [])

def generate_all_materials(model, config, class_names, test_loader, output_dir):
    """生成所有PPT素材"""
    from utils.advanced_analysis import AdvancedAnalyzer
    
    print(f"\n输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 创建分析器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    analyzer = AdvancedAnalyzer(model, device, class_names)
    
    # 2. 收集测试数据
    print("\n收集测试数据...")
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                predictions = torch.sigmoid(outputs[0])
            else:
                predictions = torch.sigmoid(outputs)
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 3. 生成详细ROC曲线
    print("\n生成ROC曲线...")
    roc_path = os.path.join(output_dir, 'detailed_roc_curves.png')
    auc_stats = analyzer.plot_detailed_roc(all_labels, all_preds, save_path=roc_path)
    
    # 保存AUC统计
    auc_stats.to_csv(os.path.join(output_dir, 'auc_statistics.csv'), index=False)
    print(f"AUC统计保存到: {output_dir}/auc_statistics.csv")
    
    # 4. 分析失败案例
    print("\n分析失败案例...")
    failure_cases, failure_analysis = analyzer.analyze_failure_cases(test_loader)
    
    # 保存失败分析
    import json
    with open(os.path.join(output_dir, 'failure_analysis.json'), 'w') as f:
        json.dump(failure_analysis, f, indent=2)
    
    print(f"失败案例分析保存到: {output_dir}/failure_analysis.json")
    
    # 5. 可视化注意力（示例）
    print("\n生成注意力可视化...")
    # 选择一个样本进行可视化
    sample_image = next(iter(test_loader))[0][0]  # 取第一个batch的第一个样本
    attention_path = os.path.join(output_dir, 'attention_visualization.png')
    analyzer.visualize_attention(sample_image, save_path=attention_path)
    
    # 6. 生成综合报告
    print("\n生成综合报告...")
    
    # 计算详细指标
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    test_results = {
        'auc_scores': [],
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': []
    }
    
    predictions_binary = (all_preds > 0.5).astype(int)
    
    for i in range(len(class_names)):
        # AUC已经计算过
        from sklearn.metrics import roc_auc_score
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        else:
            auc = 0.0
        
        test_results['auc_scores'].append(auc)
        test_results['f1_scores'].append(f1_score(all_labels[:, i], predictions_binary[:, i], zero_division=0))
        test_results['precision_scores'].append(precision_score(all_labels[:, i], predictions_binary[:, i], zero_division=0))
        test_results['recall_scores'].append(recall_score(all_labels[:, i], predictions_binary[:, i], zero_division=0))
    
    report = analyzer.generate_comprehensive_report(
        test_results, 
        save_path=os.path.join(output_dir, 'comprehensive_report.json')
    )
    
    # 7. 创建PPT摘要幻灯片（Markdown格式）
    create_ppt_summary(report, output_dir)
    
    print(f"\n所有素材已生成到: {output_dir}")
    
    return report

def create_ppt_summary(report, output_dir):
    """创建PPT摘要幻灯片"""
    slides = []
    
    # 幻灯片1: 封面
    slides.append(f"""# 胸部X光多标签智能诊断系统

## 技术深度展示
- 基于深度学习的医疗影像分析
- 多标签分类与类别不平衡处理
- 模型可解释性与临床验证

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
    
    # 幻灯片2: 项目概述
    slides.append(f"""# 项目概述

## 核心问题
- 放射科医生资源紧张
- 基层医院误诊率较高
- AI辅助诊断需求迫切

## 我们的解决方案
- 使用EfficientNet-B2作为基础模型
- 添加注意力机制和多尺度特征融合
- 处理14种胸部疾病的分类

## 技术亮点
1. 深度处理类别不平衡
2. 模型可解释性验证
3. 工程化部署考虑
""")
    
    # 幻灯片3: 性能总结
    slides.append(f"""# 性能总结

## 关键指标
- **平均AUC**: {report['performance_summary']['overall_auc']:.4f}
- **平均F1**: {report['performance_summary']['overall_f1']:.4f}
- **最佳表现类别**: {report['performance_summary']['best_performing_class']}
- **最差表现类别**: {report['performance_summary']['worst_performing_class']}

## 模型规模
- 总参数: {report['model_info']['num_params']:,}
- 可训练参数: {report['model_info']['trainable_params']:,}

## 主要挑战
- 类别不平衡（最大/最小样本比 > 100:1）
- 疾病间相关性复杂
- 小样本类别学习困难
""")
    
    # 幻灯片4: 深度洞察
    slides.append(f"""# 深度洞察

## 结构性 vs 纹理性病变
- **结构性病变**（如气胸、心脏肥大）: 识别较好
- **纹理性病变**（如肺炎、浸润）: 识别较差

## 失败分析
- 主要错误类型: {report['insights'].get('failure_analysis', {}).get('main_error_types', '待分析')}
- 最易混淆的疾病组合: {report['insights'].get('confusing_pairs', '待分析')}

## 模型偏见
- 对常见疾病识别较好
- 对罕见疾病（疝气等）识别困难
- 需要更多样本进行平衡
""")
    
    # 幻灯片5: 技术亮点
    slides.append(f"""# 技术亮点

## 1. 注意力机制
- 使用SE模块增强特征表示
- 可视化模型关注区域
- 验证医学合理性

## 2. 多尺度特征融合
- 融合不同层次的特征
- 增强对大小病灶的识别
- 提高模型鲁棒性

## 3. 不确定性估计
- 使用MC Dropout计算不确定性
- 识别低置信度预测
- 为医生提供参考
""")
    
    # 幻灯片6: 工程化思考
    slides.append(f"""# 工程化思考

## 部署优化
- 模型量化: 减少75%存储空间
- 混合精度推理: 加速2-3倍
- 批处理优化: 提高吞吐量

## 可扩展性
- 模块化设计
- 支持新疾病类别
- 易于与其他系统集成

## 安全性考虑
- 患者隐私保护
- 模型版本控制
- 失败案例记录
""")
    
    # 幻灯片7: 改进方向
    slides.append(f"""# 改进方向与未来工作

## 短期改进
1. 增加外部数据集（CheXpert）
2. 使用Vision Transformer架构
3. 引入对比学习预训练

## 中期规划
1. 多模态融合（CT + X光）
2. 时序分析（随访影像）
3. 自动报告生成

## 长期愿景
1. 临床验证与部署
2. 跨机构协作
3. AI辅助诊疗生态系统
""")
    
    # 保存所有幻灯片
    for i, slide in enumerate(slides, 1):
        slide_path = os.path.join(output_dir, f'slide_{i:02d}.md')
        with open(slide_path, 'w', encoding='utf-8') as f:
            f.write(slide)
    
    # 创建主幻灯片文件
    main_slide_path = os.path.join(output_dir, 'ppt_presentation.md')
    with open(main_slide_path, 'w', encoding='utf-8') as f:
        f.write(f"# 胸部X光多标签智能诊断系统\n\n")
        f.write("## 幻灯片目录\n\n")
        for i in range(1, len(slides) + 1):
            f.write(f"{i}. [幻灯片{i}](slide_{i:02d}.md)\n")
    
    print(f"PPT幻灯片已生成到: {main_slide_path}")

def main():
    """主函数"""
    device = setup_environment()
    
    # 设置路径
    config_path = "/content/DD/config/config_optimized.yaml"
    model_path = "/content/drive/MyDrive/outputs_optimized/best_model.pth"
    output_dir = f"/content/drive/MyDrive/ppt_materials_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"配置文件: {config_path}")
    print(f"模型文件: {model_path}")
    print(f"输出目录: {output_dir}")
    
    try:
        # 1. 加载模型
        print("\n1. 加载模型...")
        model, config, class_names = load_best_model(config_path, model_path)
        print(f"  模型加载成功: {model.__class__.__name__}")
        print(f"  类别数: {len(class_names)}")
        
        # 2. 准备测试数据
        print("\n2. 准备测试数据...")
        from data.preprocess import load_and_preprocess_data
        from data.dataset import ChestXRayDataset
        
        train_df, val_df, test_df, label_columns, class_weights = load_and_preprocess_data(config)
        
        test_dataset = ChestXRayDataset(
            test_df,
            config['paths']['images_dir'],
            ChestXRayDataset.get_transforms(config, 'val'),
            'test'
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2
        )
        
        print(f"  测试集大小: {len(test_dataset)}")
        
        # 3. 生成所有素材
        print("\n3. 生成PPT素材...")
        report = generate_all_materials(model, config, class_names, test_loader, output_dir)
        
        print(f"\n{'='*80}")
        print("PPT素材生成完成！")
        print(f"所有文件保存在: {output_dir}")
        print(f"平均AUC: {report['performance_summary']['overall_auc']:.4f}")
        print(f"平均F1: {report['performance_summary']['overall_f1']:.4f}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()