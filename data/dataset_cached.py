# /content/DD/data/dataset_cached.py
import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

class CachedChestXRayDataset(Dataset):
    """带缓存的高效数据集"""
    
    def __init__(self, df, images_dir, transform=None, phase='train', 
                 image_size=224, cache_dir=None, use_cache=True):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.phase = phase
        self.image_size = image_size
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # 只保留必要的列
        self.label_columns = [col for col in df.columns if col not in ['Image Index', 'image_path']]
        
        # 创建缓存目录
        if cache_dir and use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_path = os.path.join(cache_dir, f"{phase}_cache.pkl")
            
            # 如果缓存存在，直接加载
            if os.path.exists(self.cache_path):
                print(f"加载缓存数据: {self.cache_path}")
                with open(self.cache_path, 'rb') as f:
                    self.cached_data = pickle.load(f)
            else:
                print(f"创建缓存数据: {self.cache_path}")
                self.cached_data = self._build_cache()
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.cached_data, f)
        else:
            self.cached_data = None
    
    def _build_cache(self):
        """构建图像和标签的缓存"""
        cache = {'image_paths': [], 'labels': []}
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"缓存{self.phase}数据"):
            image_name = row['Image Index']
            
            # 查找图像路径
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                test_path = os.path.join(self.images_dir, image_name + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break
            
            if img_path and os.path.exists(img_path):
                try:
                    # 快速加载并调整大小
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.image_size, self.image_size))
                        
                        cache['image_paths'].append(img_path)
                        cache['labels'].append(row[self.label_columns].values.astype(np.float32))
                except Exception as e:
                    print(f"无法加载图像 {img_path}: {e}")
        
        return cache
    
    def __len__(self):
        if self.cached_data:
            return len(self.cached_data['labels'])
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.cached_data:
            # 从缓存获取
            img_path = self.cached_data['image_paths'][idx]
            labels = self.cached_data['labels'][idx]
            
            # 加载图像（已调整大小）
            img = Image.fromarray(self._load_cached_image(idx))
        else:
            # 原始加载方式
            row = self.df.iloc[idx]
            img_path = self._find_image_path(row['Image Index'])
            labels = row[self.label_columns].values.astype(np.float32)
            
            if img_path:
                img = Image.open(img_path).convert('RGB')
            else:
                img = Image.new('RGB', (self.image_size, self.image_size), color=0)
        
        if self.transform:
            img = self.transform(img)
        
        return img, labels
    
    def _load_cached_image(self, idx):
        """从磁盘重新加载图像（用于变换）"""
        img_path = self.cached_data['image_paths'][idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        return img
    
    def _find_image_path(self, image_name):
        """查找图像路径"""
        for ext in ['.png', '.jpg', '.jpeg']:
            test_path = os.path.join(self.images_dir, image_name + ext)
            if os.path.exists(test_path):
                return test_path
        return None