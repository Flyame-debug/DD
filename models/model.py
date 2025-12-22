import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from typing import Optional, List

class MultiLabelModel(nn.Module):
    """
    多标签分类模型，基于预训练的特征提取器
    支持 DenseNet 和 EfficientNet 系列
    """
    
    def __init__(self, base_model: str = 'densenet121', 
                 num_classes: int = 14,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 feature_dim: Optional[int] = None):
        super(MultiLabelModel, self).__init__()
        
        self.base_model_name = base_model
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        
        # 创建基础模型
        if base_model.startswith('densenet'):
            self.backbone, feature_dim = self._create_densenet(base_model, pretrained)
        elif base_model.startswith('efficientnet'):
            self.backbone, feature_dim = self._create_efficientnet(base_model, pretrained)
        else:
            raise ValueError(f"不支持的模型类型: {base_model}")
        
        # 添加自定义的分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        # 初始化分类头权重
        self._initialize_classifier()
        
    def _create_densenet(self, model_name: str, pretrained: bool):
        """创建 DenseNet 模型"""
        model_map = {
            'densenet121': models.densenet121,
            'densenet169': models.densenet169,
            'densenet201': models.densenet201,
        }
        
        if model_name not in model_map:
            raise ValueError(f"不支持的 DenseNet 模型: {model_name}")
        
        # 加载预训练模型
        model = model_map[model_name](pretrained=pretrained)
        
        # 获取特征维度
        feature_dim = model.classifier.in_features
        
        # 移除原始分类器
        model.classifier = nn.Identity()
        
        return model, feature_dim
    
    def _create_efficientnet(self, model_name: str, pretrained: bool):
        """创建 EfficientNet 模型"""
        if pretrained:
            model = EfficientNet.from_pretrained(model_name)
        else:
            model = EfficientNet.from_name(model_name)
        
        # 获取特征维度
        feature_dim = model._fc.in_features
        
        # 移除原始分类器
        model._fc = nn.Identity()
        
        return model, feature_dim
    
    def _initialize_classifier(self):
        """初始化分类头权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 提取特征
        features = self.backbone(x)
        # 分类
        logits = self.classifier(features)
        return logits

def DenseNet121MultiLabel(num_classes: int = 14, pretrained: bool = True, dropout_rate: float = 0.5):
    """创建基于 DenseNet121 的多标签分类模型"""
    return MultiLabelModel(
        base_model='densenet121',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )

def EfficientNetB4MultiLabel(num_classes: int = 14, pretrained: bool = True, dropout_rate: float = 0.5):
    """创建基于 EfficientNet-B4 的多标签分类模型"""
    return MultiLabelModel(
        base_model='efficientnet-b4',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
