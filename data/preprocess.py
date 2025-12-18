import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

def load_and_preprocess_data(config):
    """加载和预处理数据"""
    
    # 读取CSV文件
    csv_path = config['paths']['csv_path']
    df = pd.read_csv(csv_path)
    
    print(f"数据集大小: {len(df)}")
    print(f"标签列: {df.columns.tolist()}")
    
    # 检查数据
    print("\n标签分布:")
    for col in df.columns[1:]:  # 跳过第一列（图像名称）
        pos_rate = df[col].mean()
        print(f"{col}: {pos_rate:.3%} ({df[col].sum()} 正样本)")
    
    # 分割数据集
    train_df, temp_df = train_test_split(
        df, test_size=config['data']['val_split'] + config['data']['test_split'],
        random_state=42, stratify=df.iloc[:, 1:]  # 根据标签分层采样
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=config['data']['test_split'] / (config['data']['val_split'] + config['data']['test_split']),
        random_state=42,
        stratify=temp_df.iloc[:, 1:]
    )
    
    print(f"\n训练集: {len(train_df)}")
    print(f"验证集: {len(val_df)}")
    print(f"测试集: {len(test_df)}")
    
    # 保存分割结果
    output_dir = config['paths']['output_dir']
    train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)
    
    # 计算类别权重（用于处理不平衡）
    class_weights = calculate_class_weights(train_df)
    
    return train_df, val_df, test_df, class_weights

def calculate_class_weights(train_df):
    """计算类别权重以处理不平衡"""
    class_weights = {}
    for i, col in enumerate(train_df.columns[1:]):
        pos = train_df[col].sum()
        neg = len(train_df) - pos
        weight_for_0 = (1 / neg) * (len(train_df) / 2.0) if neg > 0 else 1.0
        weight_for_1 = (1 / pos) * (len(train_df) / 2.0) if pos > 0 else 1.0
        class_weights[i] = torch.tensor([weight_for_0, weight_for_1])
    
    return class_weights