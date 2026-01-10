# /content/DD/data/dataset_cached.py
import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import glob

class CachedChestXRayDataset(Dataset):
    """带缓存的高效数据集 - 修复版本"""
    
    def __init__(self, df, images_dir, transform=None, phase='train', 
                 image_size=224, cache_dir=None, use_cache=True):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.phase = phase
        self.image_size = image_size
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # 识别标签列
        self.label_columns = self._identify_label_columns(df)
        print(f"识别到 {len(self.label_columns)} 个标签列: {self.label_columns}")
        
        # 收集有效的样本
        self.valid_indices, self.image_paths = self._collect_valid_samples()
        
        print(f"{phase}数据集: {len(self.valid_indices)} 个有效样本")
        
        if len(self.valid_indices) == 0:
            raise ValueError(f"没有找到任何有效的样本！phase={phase}")
    
    def _identify_label_columns(self, df):
        """识别标签列"""
        # 排除非标签列
        non_label_cols = ['Image Index', 'image_path', 'Finding Labels', 'Follow-up #', 
                         'Patient ID', 'Patient Age', 'Patient Gender', 'View Position',
                         'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]']
        
        label_columns = []
        for col in df.columns:
            if col not in non_label_cols:
                label_columns.append(col)
        
        return label_columns
    
    def _collect_valid_samples(self):
        """收集有效样本"""
        valid_indices = []
        image_paths = []
        
        print(f"开始收集{self.phase}数据集的有效样本...")
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"处理{self.phase}数据"):
            image_name = row['Image Index']
            img_path = self._find_image_path(image_name)
            
            if img_path and os.path.exists(img_path):
                try:
                    # 验证图像是否可以打开
                    with Image.open(img_path) as img:
                        img.verify()  # 验证文件完整性
                    valid_indices.append(idx)
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"警告: 图像 {img_path} 损坏: {e}")
        
        return valid_indices, image_paths
    
    def _find_image_path(self, image_name):
        """查找图像路径"""
        # 移除可能的扩展名
        base_name = os.path.splitext(image_name)[0]
        
        # 首先尝试直接匹配
        possible_paths = [
            os.path.join(self.images_dir, image_name),
            os.path.join(self.images_dir, base_name + '.png'),
            os.path.join(self.images_dir, base_name + '.jpg'),
            os.path.join(self.images_dir, base_name + '.jpeg'),
            os.path.join(self.images_dir, base_name + '.PNG'),
            os.path.join(self.images_dir, base_name + '.JPG'),
            os.path.join(self.images_dir, base_name + '.JPEG'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果没找到，尝试在子目录中查找
        for root, dirs, files in os.walk(self.images_dir):
            for file in files:
                if file.startswith(base_name):
                    return os.path.join(root, file)
        
        return None
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # 获取原始索引
        original_idx = self.valid_indices[idx]
        img_path = self.image_paths[idx]
        
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {img_path}: {e}")
            # 创建黑色图像作为替代
            img = Image.new('RGB', (self.image_size, self.image_size), color=0)
        
        # 获取标签
        row = self.df.iloc[original_idx]
        labels = []
        for col in self.label_columns:
            if col in row:
                labels.append(float(row[col]))
            else:
                labels.append(0.0)
        
        labels = np.array(labels, dtype=np.float32)
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        return img, labels

# 为了方便，也创建一个快速加载版本
class FastChestXRayDataset(Dataset):
    """快速加载的数据集，不使用缓存"""
    
    def __init__(self, df, images_dir, transform=None, image_size=224):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.image_size = image_size
        
        # 识别标签列
        non_label_cols = ['Image Index', 'image_path', 'Finding Labels']
        self.label_columns = [col for col in df.columns if col not in non_label_cols]
        
        # 预计算图像路径
        self.image_paths = []
        self.valid_indices = []
        
        print(f"准备快速数据集，共有 {len(df)} 个样本")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="准备数据"):
            image_name = row['Image Index']
            img_path = self._find_image_path(image_name)
            
            if img_path and os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.valid_indices.append(idx)
        
        print(f"找到 {len(self.valid_indices)} 个有效样本")
    
    def _find_image_path(self, image_name):
        """查找图像路径"""
        base_name = os.path.splitext(image_name)[0]
        
        # 尝试常见扩展名
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            path = os.path.join(self.images_dir, base_name + ext)
            if os.path.exists(path):
                return path
        
        # 尝试直接使用原文件名
        path = os.path.join(self.images_dir, image_name)
        if os.path.exists(path):
            return path
        
        return None
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        img_path = self.image_paths[idx]
        
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (self.image_size, self.image_size), color=0)
        
        # 获取标签
        row = self.df.iloc[original_idx]
        labels = []
        for col in self.label_columns:
            if col in row:
                labels.append(float(row[col]))
            else:
                labels.append(0.0)
        
        labels = np.array(labels, dtype=np.float32)
        
        if self.transform:
            img = self.transform(img)
        
        return img, labels