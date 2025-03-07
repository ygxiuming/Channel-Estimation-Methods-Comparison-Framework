"""
信道估计训练器模块

此模块实现了用于训练深度学习模型的训练器类。训练器提供了完整的训练流程管理，
包括训练循环、验证评估、模型保存、早停机制等功能。

主要特性：
- 支持GPU训练
- 实时进度显示
- TensorBoard可视化
- 自动保存最佳模型
- 训练状态恢复
- 早停机制
- 学习率自适应调整

作者: lzm
日期: 2025-03-07
"""

import os
import time
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

class ChannelEstimatorTrainer:
    """
    信道估计模型训练器
    
    该类实现了完整的模型训练流程，包括：
    1. 训练和验证循环
    2. 损失计算和优化
    3. 进度监控和可视化
    4. 模型保存和加载
    5. 训练状态管理
    
    参数:
        model (nn.Module): 要训练的模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        save_dir (str): 模型和日志保存目录
        project_name (str): 项目名称，用于创建子目录
        device (str): 训练设备（'cuda'或'cpu'）
        learning_rate (float): 初始学习率
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        save_dir='runs/train',
        project_name='channel_estimation',
        device='cuda',
        learning_rate=0.001
    ):
        # 初始化模型和数据加载器
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        
        # 创建保存目录
        self.save_dir = Path(save_dir) / project_name
        self.weights_dir = self.save_dir / 'weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置TensorBoard
        self.writer = SummaryWriter(str(self.save_dir / 'tensorboard'))
        
        # 初始化优化器和损失函数
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 设置学习率调度器，当验证损失不再下降时降低学习率
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, verbose=True
        )
        
        # 初始化训练状态
        self.best_val_loss = float('inf')  # 最佳验证损失
        self.current_epoch = 0             # 当前训练轮数
        self.history = {                   # 训练历史记录
            'train_loss': [],             # 训练损失历史
            'val_loss': [],               # 验证损失历史
            'learning_rate': []           # 学习率历史
        }
        
    def train_epoch(self):
        """
        训练一个epoch
        
        该方法实现了单个训练epoch的完整流程：
        1. 遍历训练数据批次
        2. 前向传播计算损失
        3. 反向传播更新参数
        4. 记录训练状态
        5. 更新进度显示
        
        返回:
            float: 该epoch的平均训练损失
        """
        self.model.train()  # 设置为训练模式
        total_loss = 0
        
        # 创建进度条
        pbar = tqdm(self.train_loader, 
                   desc=f'Epoch {self.current_epoch + 1:<3d}', 
                   leave=True,  # 保持显示
                   position=1,   # 位置在总进度条下方
                   bar_format='{desc:<12} {percentage:3.0f}%|{bar:50}{r_bar}',
                   ncols=120)
        
        # 遍历训练数据批次
        for batch_idx, (X, y) in enumerate(pbar):
            # 将数据移到指定设备
            X, y = X.to(self.device), y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()  # 清除梯度
            outputs = self.model(X)     # 模型预测
            loss = self.criterion(outputs, y)  # 计算损失
            
            # 反向传播
            loss.backward()         # 计算梯度
            self.optimizer.step()   # 更新参数
            
            # 更新损失统计
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # 更新进度条显示
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'}, refresh=True)
            
            # 记录到TensorBoard
            step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/batch_loss', loss.item(), step)
        
        pbar.close()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """
        验证模型性能
        
        该方法在验证集上评估模型性能：
        1. 不计算梯度
        2. 遍历验证数据批次
        3. 计算验证损失
        4. 更新进度显示
        
        返回:
            float: 验证集上的平均损失
        """
        self.model.eval()  # 设置为评估模式
        total_loss = 0
        val_loader_len = len(self.val_loader)
        
        with torch.no_grad():  # 不计算梯度
            # 创建进度条
            pbar = tqdm(self.val_loader, 
                       desc='Validating', 
                       leave=True,  # 保持显示
                       position=1,   # 位置在总进度条下方
                       bar_format='{desc:<12} {percentage:3.0f}%|{bar:50}{r_bar}',
                       ncols=120)
            
            # 遍历验证数据批次
            for batch_idx, (X, y) in enumerate(pbar):
                # 将数据移到指定设备
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)     # 模型预测
                loss = self.criterion(outputs, y)  # 计算损失
                total_loss += loss.item()
                
                # 更新进度条显示
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.6f}'}, refresh=True)
            
            pbar.close()
        
        return total_loss / val_loader_len
    
    def train(self, epochs=100, early_stopping_patience=10):
        """
        完整的训练流程
        
        该方法实现了完整的模型训练流程：
        1. 多轮训练循环
        2. 定期验证评估
        3. 模型保存
        4. 早停机制
        5. 进度监控
        
        参数:
            epochs (int): 训练轮数
            early_stopping_patience (int): 早停耐心值，验证损失多少轮未改善时停止训练
        """
        early_stopping_counter = 0
        start_time = time.time()
        
        # 打印训练配置
        print("\n训练配置:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {self.train_loader.batch_size}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Device: {self.device}")
        print(f"  Early Stopping Patience: {early_stopping_patience}")
        print()
        
        # 创建总进度条
        pbar = tqdm(range(epochs), 
                   desc='Progress', 
                   leave=True,     # 保持显示
                   position=0,     # 位置在最上方
                   bar_format='{desc:<12} {percentage:3.0f}%|{bar:50}{r_bar}',
                   ncols=120)
        
        try:
            # 训练循环
            for epoch in pbar:
                self.current_epoch = epoch
                
                # 训练一个epoch
                train_loss = self.train_epoch()
                
                # 验证评估
                val_loss = self.validate()
                
                # 更新学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)  # 根据验证损失调整学习率
                
                # 记录训练历史
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['learning_rate'].append(current_lr)
                
                # 记录到TensorBoard
                self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
                self.writer.add_scalar('val/epoch_loss', val_loss, epoch)
                self.writer.add_scalar('train/lr', current_lr, epoch)
                
                # 更新总进度条显示
                pbar.set_postfix({
                    'train': f'{train_loss:.6f}',
                    'val': f'{val_loss:.6f}',
                    'lr': f'{current_lr:.2e}'
                }, refresh=True)
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model('best.pt')
                    early_stopping_counter = 0  # 重置早停计数器
                else:
                    early_stopping_counter += 1
                
                # 定期保存模型
                if (epoch + 1) % 10 == 0:
                    self.save_model(f'epoch_{epoch + 1}.pt')
                
                # 早停检查
                if early_stopping_counter >= early_stopping_patience:
                    print(f'\nEarly stopping triggered at epoch {epoch + 1}')
                    break
                
                # 在每个epoch结束时打印空行
                print()
        
        except KeyboardInterrupt:
            print('\nTraining interrupted by user')
        
        finally:
            # 保存最后的模型和清理资源
            self.save_model('last.pt')
            self.writer.close()
            
            # 打印训练结果统计
            elapsed_time = time.time() - start_time
            print("\n训练完成:")
            print(f"  训练轮数: {self.current_epoch + 1}/{epochs}")
            print(f"  最佳验证损失: {self.best_val_loss:.6f}")
            print(f"  训练时间: {elapsed_time/3600:.2f}h")
            print(f"  保存路径: {self.weights_dir}")
            print()
    
    def save_model(self, filename):
        """
        保存模型状态
        
        保存完整的训练状态，包括：
        - 模型参数
        - 优化器状态
        - 学习率调度器状态
        - 训练历史
        - 最佳验证损失
        
        参数:
            filename (str): 保存的文件名
        """
        save_path = self.weights_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, str(save_path))
    
    def load_model(self, filename):
        """
        加载模型状态
        
        加载完整的训练状态，包括：
        - 模型参数
        - 优化器状态
        - 学习率调度器状态
        - 训练历史
        - 最佳验证损失
        
        参数:
            filename (str): 要加载的文件名
            
        返回:
            bool: 是否成功加载模型
        """
        load_path = self.weights_dir / filename
        if load_path.exists():
            checkpoint = torch.load(str(load_path))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            if 'history' in checkpoint:
                self.history = checkpoint['history']
            return True
        return False
    
    def get_history(self):
        """
        获取训练历史
        
        返回:
            dict: 包含训练损失、验证损失和学习率的历史记录
        """
        return self.history 