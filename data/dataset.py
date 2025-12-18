import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ChestXRayDataset(Dataset):
    """胸部X光多标签数据集类"""
    
    def __init__(self, dataframe, image_dir, transform=None, phase='train'):
        """
        Args:
            dataframe: 包含图像路径和标签的DataFrame
            image_dir: 图像目录
            transform: 数据增强转换
            phase: 'train', 'val', 或 'test'
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.phase = phase
        self.transform = transform
        self.class_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # 获取图像路径和标签
        img_name = self.dataframe.iloc[idx]['Image Index']
        img_path = os.path.join(self.image_dir, img_name)
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # 获取14个标签（假设DataFrame中有对应列）
        labels = self.dataframe.iloc[idx][self.class_names].values.astype(np.float32)
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, torch.FloatTensor(labels)
    
    @staticmethod
    def get_transforms(config, phase='train'):
        """获取数据增强转换"""
        image_size = config['data']['image_size']
        
        if phase == 'train':
            transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=config['data']['mean'],
                    std=config['data']['std'],
                ),
                ToTensorV2(),
            ])
        else:
            transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(
                    mean=config['data']['mean'],
                    std=config['data']['std'],
                ),
                ToTensorV2(),
            ])
        
        return transform