import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DLChannelEstimator(nn.Module):
    """基于深度学习的信道估计器"""
    
    def __init__(self, input_size, hidden_size=128):
        super(DLChannelEstimator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, input_size)
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = None
        
    def forward(self, x):
        """前向传播"""
        return self.network(x)
    
    def train_model(self, train_loader, epochs=100, learning_rate=0.001):
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
        """
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_Y in train_loader:
                self.optimizer.zero_grad()
                
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_Y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    def estimate(self, X):
        """
        使用训练好的模型进行信道估计
        
        参数:
            X: 输入信号
            
        返回:
            估计的信道响应
        """
        self.eval()
        with torch.no_grad():
            return self(X)
            
    def get_mse(self, y_true, y_pred):
        """计算均方误差"""
        return self.criterion(y_pred, y_true).item() 