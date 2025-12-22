import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast
import json

def load_and_preprocess_data(config):
    """
    加载CSV数据，将文本标签转换为多热编码，并划分数据集。
    
    Args:
        config: 配置字典
        
    Returns:
        train_df, val_df, test_df: 划分后的DataFrame（包含图像路径和多热编码标签）
        label_columns: 疾病标签的列名列表
        (可选) class_weights: 类别权重列表
    """
    csv_path = config['paths']['csv_path']
    images_dir = config['paths']['images_dir']
    
    # 1. 加载数据
    print(f"正在加载数据从: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"数据集大小: {len(df)}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 2. 确定标签列名（根据你的CSV文件调整）
    # 假设你的标签列名为 'Finding Labels'，内容像是 "Cardiomegaly|Effusion"
    label_column_name = 'Finding Labels'  # 可能需要修改
    
    if label_column_name not in df.columns:
        # 尝试寻找可能的标签列
        possible_label_cols = [col for col in df.columns if 'label' in col.lower() or 'finding' in col.lower()]
        if possible_label_cols:
            label_column_name = possible_label_cols[0]
            print(f"警告: 未找到 'Finding Labels' 列，改用 '{label_column_name}'")
        else:
            raise ValueError(f"在CSV文件中未找到标签列。现有列: {df.columns.tolist()}")
    
    # 3. 提取所有唯一的疾病类别
    # 假设标签格式是用 '|' 分隔的字符串，例如 "Cardiomegaly|Effusion"
    all_labels = set()
    for labels_str in df[label_column_name].fillna('').astype(str):
        # 分割字符串，去除空白字符
        labels = [lb.strip() for lb in labels_str.split('|') if lb.strip()]
        all_labels.update(labels)
    
    # 转换为排序后的列表，确保顺序一致
    label_columns = sorted(list(all_labels))
    print(f"发现 {len(label_columns)} 种疾病类别: {label_columns}")
    
    # 4. 将文本标签转换为多热编码 (0/1矩阵)
    print("正在将文本标签转换为多热编码...")
    for disease in label_columns:
        # 如果某行的标签字符串中包含该疾病名，则标记为1，否则为0
        df[disease] = df[label_column_name].fillna('').astype(str).apply(
            lambda x: 1 if disease in [lb.strip() for lb in x.split('|')] else 0
        )
    
    # 5. 添加图像完整路径列（假设CSV中有 'Image Index' 列）
    if 'Image Index' in df.columns:
        df['image_path'] = df['Image Index'].apply(lambda x: f"{images_dir}/{x}")
    else:
        # 如果没有，尝试第一列作为文件名
        first_col = df.columns[0]
        df['image_path'] = df[first_col].apply(lambda x: f"{images_dir}/{x}")
    
    # 6. 计算类别权重（用于处理不平衡数据，可选）
    # 类别权重与正样本数量成反比，帮助模型关注罕见病
    print("计算类别权重...")
    class_weights = []
    for disease in label_columns:
        pos_count = df[disease].sum()
        total = len(df)
        # 避免除零，使用平滑处理
        weight = total / (2.0 * pos_count + 1e-5)
        class_weights.append(weight)
        pos_rate = pos_count / total
        print(f"  {disease}: 正样本数={pos_count} ({pos_rate:.2%}), 权重={weight:.2f}")
    
    # 归一化权重
    class_weights = np.array(class_weights)
    class_weights = class_weights / class_weights.mean()  # 使平均权重为1
    
            # 7. 数据集划分 (为不平衡的多标签数据采用简化策略)
    train_ratio = config['data'].get('train_split', 0.7)
    val_ratio = config['data'].get('val_split', 0.15)
    # test_ratio 可通过 1 - train_ratio - val_ratio 得到
    
    print("正在进行数据集划分（使用随机策略）...")
    
    # 第一步：划分训练集和临时集（测试集+验证集）
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),  # 临时集占30%
        random_state=42,              # 固定随机种子保证结果可复现
        # 关键：移除 stratify 参数，进行随机划分
        # stratify=df[label_columns].values  # 注释掉或删除这行
    )
    
    # 第二步：将临时集划分为验证集和测试集
    val_test_ratio = val_ratio / (val_ratio + config['data'].get('test_split', 0.15))
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_test_ratio), # 测试集在临时集中的比例
        random_state=42,
        # 同样，移除分层参数
        # stratify=temp_df[label_columns].values  # 注释掉或删除这行
    )
    
    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_df)} 个样本 ({len(train_df)/len(df):.1%})")
    print(f"  验证集: {len(val_df)} 个样本 ({len(val_df)/len(df):.1%})")
    print(f"  测试集: {len(test_df)} 个样本 ({len(test_df)/len(df):.1%})")
    
    # （可选）验证划分后各类别的样本数，确保没有类别在某个集中完全消失
    print("\n划分后各类别正样本数统计（前5类）:")
    for i, disease in enumerate(label_columns[:5]):  # 查看前5类
        train_pos = train_df[disease].sum()
        val_pos = val_df[disease].sum()
        test_pos = test_df[disease].sum()
        print(f"  {disease:20} 训练集: {train_pos:4d}, 验证集: {val_pos:4d}, 测试集: {test_pos:4d}")
    # 8. 返回结果（根据你 train.py 中的调用方式调整）
    # 通常需要返回划分后的DataFrame和标签列名
    return train_df, val_df, test_df, label_columns, class_weights