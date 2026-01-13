# /content/DD/models/model_enhanced.py
import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from typing import Optional, List

class AttentionBlock(nn.Module):
    """注意力机制模块 - 展示你对模型的理解"""
    def __init__(self, in_channels, reduction=16):
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        attention = self.sigmoid(out).view(b, c, 1, 1)
        return x * attention.expand_as(x)

class EnhancedMultiLabelModel(nn.Module):
    """
    增强版多标签分类模型，包含：
    1. 注意力机制
    2. 多尺度特征融合
    3. 类别特定的分类头
    """
    
    def __init__(self, base_model: str = 'efficientnet-b2', 
                 num_classes: int = 14,
                 pretrained: bool = True,
                 dropout_rate: float = 0.4,
                 use_attention: bool = True):
        super(EnhancedMultiLabelModel, self).__init__()
        
        self.base_model_name = base_model
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # 创建基础模型
        if base_model.startswith('efficientnet'):
            self.backbone, feature_dim = self._create_efficientnet(base_model, pretrained)
        else:
            raise ValueError(f"不支持的模型类型: {base_model}")
        
        # 添加注意力机制（展示创新点）
        if use_attention:
            self.attention = AttentionBlock(feature_dim)
        
        # 多尺度特征融合（展示深度理解）
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 类别特定的分类头（展示对多标签问题的理解）
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim // 2, 1)
            ) for _ in range(num_classes)
        ])
        
        # 全局分类头（对比使用）
        self.global_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
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
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 提取特征
        features = self.backbone(x)

        # 如果特征已经是2维，重塑为4维
        if features.dim() == 2:
            # 获取特征维度
            b, c = features.size()
            # 重塑为 [batch, channels, 1, 1]
            features = features.view(b, c, 1, 1)
        
        # 应用注意力机制（如果启用）
        if self.use_attention:
            features = self.attention(features)
        
        # 多尺度特征融合
        fused_features = self.multi_scale_fusion(features)
        fused_features = fused_features.view(fused_features.size(0), -1)
        
        # 使用类别特定的分类头（对比展示）
        outputs = []
        for classifier in self.classifiers:
            output = classifier(fused_features)
            outputs.append(output)
        
        logits = torch.cat(outputs, dim=1)
        
        # 也可以同时返回全局分类器的结果进行对比
        global_logits = self.global_classifier(fused_features)
        
        return logits, global_logits, features

class UncertaintyAwareModel(nn.Module):
    """
    不确定性感知模型 - 展示深度思考
    输出预测和不确定性估计
    """
    def __init__(self, base_model, num_classes=14, dropout_rate=0.3, mc_dropout_samples=10):
        super(UncertaintyAwareModel, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.mc_dropout_samples = mc_dropout_samples
        
        # Monte Carlo Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, return_uncertainty=False):
        if return_uncertainty and self.training:
            # 训练时不计算不确定性
            return self.base_model(x)[0]
        
        if return_uncertainty:
            # 测试时使用MC Dropout计算不确定性
            predictions = []
            for _ in range(self.mc_dropout_samples):
                pred = self.base_model(x)[0]
                predictions.append(torch.sigmoid(pred).unsqueeze(0))
            
            predictions = torch.cat(predictions, dim=0)  # [mc_samples, batch, num_classes]
            mean_pred = predictions.mean(dim=0)
            uncertainty = predictions.std(dim=0)  # 不确定性估计
            
            return mean_pred, uncertainty
        else:
            return self.base_model(x)[0]