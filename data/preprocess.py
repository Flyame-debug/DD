import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast
import json
import os
from pathlib import Path

def load_and_preprocess_data(config):
    """
    加载CSV数据，将文本标签转换为多热编码，并划分数据集。
    
    Args:
        config: 配置字典
        
    Returns:
        train_df, val_df, test_df: 划分后的DataFrame（包含图像路径和多热编码标签）
        label_columns: 疾病标签的列名列表
        class_weights: 类别权重列表
    """
    csv_path = config['paths']['csv_path']
    images_dir = config['paths']['images_dir']
    
    # 1. 加载数据
    print(f"正在加载数据从: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"原始数据集大小: {len(df)}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 2. 确定标签列名
    label_column_name = 'Finding Labels'
    
    if label_column_name not in df.columns:
        possible_label_cols = [col for col in df.columns if 'label' in col.lower() or 'finding' in col.lower()]
        if possible_label_cols:
            label_column_name = possible_label_cols[0]
            print(f"警告: 未找到 'Finding Labels' 列，改用 '{label_column_name}'")
        else:
            raise ValueError(f"在CSV文件中未找到标签列。现有列: {df.columns.tolist()}")
    
    # 3. 获取实际存在的图片文件
    print("正在扫描图片目录...")
    existing_images = set()
    image_path_map = {}
    
    # 递归搜索所有图片文件
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                existing_images.add(file)
                # 记录完整路径
                image_path_map[file] = os.path.join(root, file)
    
    print(f"找到 {len(existing_images)} 个图片文件")
    
    # 4. 过滤掉不存在的图片
    initial_count = len(df)
    df = df[df['Image Index'].isin(existing_images)].copy()
    filtered_count = len(df)
    print(f"过滤后数据集大小: {filtered_count} ({filtered_count/initial_count*100:.1f}%)")
    
    if filtered_count == 0:
        raise ValueError("错误: 没有找到任何存在的图片文件！请检查图片路径和文件名。")
    
    # 5. 添加完整的图片路径
    df['image_path'] = df['Image Index'].apply(lambda x: image_path_map.get(x, ''))
    
    # 6. 提取所有唯一的疾病类别
    all_labels = set()
    for labels_str in df[label_column_name].fillna('').astype(str):
        # 分割字符串，去除空白字符
        labels = [lb.strip() for lb in labels_str.split('|') if lb.strip()]
        all_labels.update(labels)
    
    # 转换为排序后的列表，确保顺序一致
    label_columns = sorted(list(all_labels))
    print(f"发现 {len(label_columns)} 种疾病类别: {label_columns}")
    
    # 7. 将文本标签转换为多热编码 (0/1矩阵)
    print("正在将文本标签转换为多热编码...")
    for disease in label_columns:
        # 如果某行的标签字符串中包含该疾病名，则标记为1，否则为0
        df[disease] = df[label_column_name].fillna('').astype(str).apply(
            lambda x: 1 if disease in [lb.strip() for lb in x.split('|')] else 0
        )
    
    # 8. 计算类别权重（用于处理不平衡数据）
    print("计算类别权重...")
    class_weights = []
    total = len(df)
    
    for i, disease in enumerate(label_columns):
        pos_count = df[disease].sum()
        neg_count = total - pos_count
        
        if pos_count > 0 and neg_count > 0:
            # 使用平衡权重：负样本数/正样本数
            weight = neg_count / pos_count
        else:
            weight = 1.0
        
        class_weights.append(weight)
        pos_rate = pos_count / total
        
        print(f"  {disease:20} 正样本数={pos_count:4d} ({pos_rate*100:5.2f}%), 权重={weight:7.2f}")
    
    # 9. 数据集划分
    train_ratio = config['data'].get('train_split', 0.7)
    val_ratio = config['data'].get('val_split', 0.15)
    test_ratio = config['data'].get('test_split', 0.15)
    
    print(f"正在进行数据集划分（训练: {train_ratio*100:.0f}%, 验证: {val_ratio*100:.0f}%, 测试: {test_ratio*100:.0f}%）...")
    
    # 第一步：划分训练集和临时集（测试集+验证集）
    temp_size = 1 - train_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=42
    )
    
    # 第二步：将临时集划分为验证集和测试集
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_test_ratio),
        random_state=42
    )
    
    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_df)} 个样本 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 个样本 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 个样本 ({len(test_df)/len(df)*100:.1f}%)")
    
    # 打印各类别分布
    print("\n划分后各类别正样本数统计:")
    for i, disease in enumerate(label_columns[:5]):  # 只显示前5个类别
        train_pos = train_df[disease].sum()
        val_pos = val_df[disease].sum()
        test_pos = test_df[disease].sum()
        print(f"  {disease:20} 训练集: {train_pos:4d}, 验证集: {val_pos:4d}, 测试集: {test_pos:4d}")
    
    # 如果有更多类别，显示总数
    if len(label_columns) > 5:
        print(f"  ... 还有 {len(label_columns)-5} 个类别")
    
    return train_df, val_df, test_df, label_columns, class_weights