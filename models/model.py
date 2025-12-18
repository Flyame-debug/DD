import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class DenseNet121MultiLabel(nn.Module):
    """基于DenseNet121的多标签分类模型"""
    
    def __init__(self, num_classes=14, pretrained=True, dropout_rate=0.5):
        super().__init__()
        
        # 加载预训练模型
        self.densenet = models.densenet121(pretrained=pretrained)
        num_features = self.densenet.classifier.in_features
        
        # 替换分类器
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # 多标签使用Sigmoid
        )
    
    def forward(self, x):
        return self.densenet(x)

class EfficientNetB4MultiLabel(nn.Module):
    """基于EfficientNet的多标签分类模型"""
    
    def __init__(self, num_classes=14, pretrained=True, dropout_rate=0.5):
        super().__init__()
        
        if pretrained:
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4')
        else:
            self.efficientnet = EfficientNet.from_name('efficientnet-b4')
        
        num_features = self.efficientnet._fc.in_features
        
        # 替换分类器
        self.efficientnet._fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.efficientnet(x)

class MultiLabelModel(nn.Module):
    """包装模型，添加辅助功能"""
    
    def __init__(self, base_model, num_classes=14):
        super().__init__()
        self.model = base_model
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.model(x)
    
    def get_feature_extractor(self):
        """获取特征提取器（用于CAM可视化）"""
        if hasattr(self.model, 'densenet'):
            # DenseNet
            modules = list(self.model.densenet.features.children())
            feature_extractor = nn.Sequential(*modules)
        elif hasattr(self.model, 'efficientnet'):
            # EfficientNet
            feature_extractor = self.model.efficientnet.extract_features
        else:
            feature_extractor = None
        
        return feature_extractor