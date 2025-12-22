import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
    """
    带类别权重的二分类交叉熵损失函数
    用于处理多标签分类中的类别不平衡问题
    """
    def __init__(self, weight=None, pos_weight=None, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, input, target):
        # 使用带logits的BCE损失，更数值稳定
        loss = F.binary_cross_entropy_with_logits(
            input, 
            target,
            weight=self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss - 专注于难分类样本的损失函数
    用于处理类别不平衡，减少易分类样本的权重
    
    参考文献: Lin et al., Focal Loss for Dense Object Detection (ICCV 2017)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        # 计算基本的BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        
        # 计算概率
        pt = torch.exp(-bce_loss)  # pt = p if y=1, else 1-p
        
        # Focal Loss公式
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(config):
    """
    根据配置文件返回损失函数
    Args:
        config: 配置字典
    Returns:
        损失函数实例
    """
    loss_name = config['training'].get('loss', 'bce')  # 默认为bce
    
    if loss_name == 'weighted_bce':
        # 如果有类别权重配置，可以在这里使用
        weight = config['training'].get('class_weights', None)
        if weight is not None:
            weight = torch.tensor(weight).float()
        return WeightedBCELoss(weight=weight)
    
    elif loss_name == 'focal':
        alpha = config['training'].get('focal_alpha', 0.25)
        gamma = config['training'].get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    else:  # 默认BCE损失
        return nn.BCEWithLogitsLoss()


# 单元测试
if __name__ == '__main__':
    # 简单测试损失函数能否正常工作
    batch_size, num_classes = 4, 14
    
    # 模拟输出和标签
    logits = torch.randn(batch_size, num_classes)
    targets = torch.rand(batch_size, num_classes).ge(0.5).float()
    
    # 测试各个损失函数
    bce_loss = WeightedBCELoss()
    print(f"WeightedBCELoss: {bce_loss(logits, targets)}")
    
    focal_loss = FocalLoss()
    print(f"FocalLoss: {focal_loss(logits, targets)}")
    
    print("All loss functions working correctly!")