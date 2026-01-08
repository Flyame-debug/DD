# /content/DD/training/trainer_optimized.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import os

class OptimizedTrainer:
    """优化后的训练器"""
    
    def __init__(self, config, device, logger):
        self.config = config
        self.device = device
        self.logger = logger
        self.scaler = GradScaler() if config['training']['use_amp'] else None
        
    def train_epoch(self, model, dataloader, criterion, optimizer, epoch):
        """混合精度训练epoch"""
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            if self.scaler:
                # 混合精度训练
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # 普通训练
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return running_loss / len(dataloader)
    
    def validate(self, model, dataloader, criterion):
        """验证"""
        model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        
        return running_loss / len(dataloader)
    
    def train(self, model, train_loader, val_loader, criterion, optimizer):
        """主训练循环"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # 验证
            val_loss = self.validate(model, val_loader, criterion)
            
            epoch_time = time.time() - start_time
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
            
            # 早停逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(model, optimizer, epoch, val_loss, 'best')
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    self.logger.info(f"早停触发，最佳Val Loss: {best_val_loss:.4f}")
                    break
            
            # 定期保存
            if (epoch + 1) % self.config['training']['save_checkpoint_freq'] == 0:
                self.save_checkpoint(model, optimizer, epoch, val_loss, f'epoch_{epoch+1}')
    
    def save_checkpoint(self, model, optimizer, epoch, val_loss, name):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        path = os.path.join(self.config['paths']['output_dir'], f'{name}.pth')
        torch.save(checkpoint, path)
        self.logger.info(f"保存检查点: {path}")