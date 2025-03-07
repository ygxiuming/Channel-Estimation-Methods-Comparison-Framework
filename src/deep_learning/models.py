import torch
import torch.nn as nn
import numpy as np

class CNNEstimator(nn.Module):
    """基于CNN的信道估计器"""
    def __init__(self, input_size, n_rx=4, n_tx=8):
        super(CNNEstimator, self).__init__()
        
        self.n_features = input_size
        self.n_rx = n_rx
        self.n_tx = n_tx
        
        self.cnn = nn.Sequential(
            # 将输入重塑为 (batch_size, 1, -1)，作为1D CNN的输入
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(64, 1, kernel_size=3, padding=1)
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_rx * n_tx)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        # 重塑为1D CNN输入
        x = x.view(batch_size, 1, -1)
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        # 重塑输出为 [batch_size, n_rx, n_tx]
        return x.view(batch_size, self.n_rx, self.n_tx)

class RNNEstimator(nn.Module):
    """基于RNN的信道估计器"""
    def __init__(self, input_size, hidden_size=256, num_layers=2, n_rx=4, n_tx=8):
        super(RNNEstimator, self).__init__()
        
        self.n_features = input_size
        self.seq_length = 8  # 将输入序列分成8个时间步
        self.feature_size = self.n_features // self.seq_length
        self.n_rx = n_rx
        self.n_tx = n_tx
        
        self.rnn = nn.RNN(self.feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_rx * n_tx)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        # 重塑为序列数据
        x = x.view(batch_size, self.seq_length, -1)
        out, _ = self.rnn(x)
        x = self.fc(out[:, -1, :])
        # 重塑输出为 [batch_size, n_rx, n_tx]
        return x.view(batch_size, self.n_rx, self.n_tx)

class LSTMEstimator(nn.Module):
    """基于LSTM的信道估计器"""
    def __init__(self, input_size, hidden_size=256, num_layers=2, n_rx=4, n_tx=8):
        super(LSTMEstimator, self).__init__()
        
        self.n_features = input_size
        self.seq_length = 8
        self.feature_size = self.n_features // self.seq_length
        self.n_rx = n_rx
        self.n_tx = n_tx
        
        self.lstm = nn.LSTM(self.feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_rx * n_tx)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_length, -1)
        out, (_, _) = self.lstm(x)
        x = self.fc(out[:, -1, :])
        # 重塑输出为 [batch_size, n_rx, n_tx]
        return x.view(batch_size, self.n_rx, self.n_tx)

class GRUEstimator(nn.Module):
    """基于GRU的信道估计器"""
    def __init__(self, input_size, hidden_size=256, num_layers=2, n_rx=4, n_tx=8):
        super(GRUEstimator, self).__init__()
        
        self.n_features = input_size
        self.seq_length = 8
        self.feature_size = self.n_features // self.seq_length
        self.n_rx = n_rx
        self.n_tx = n_tx
        
        self.gru = nn.GRU(self.feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_rx * n_tx)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_length, -1)
        out, _ = self.gru(x)
        x = self.fc(out[:, -1, :])
        # 重塑输出为 [batch_size, n_rx, n_tx]
        return x.view(batch_size, self.n_rx, self.n_tx)

class HybridEstimator(nn.Module):
    """混合深度学习估计器（CNN+LSTM）"""
    def __init__(self, input_size, hidden_size=256, n_rx=4, n_tx=8):
        super(HybridEstimator, self).__init__()
        
        self.n_features = input_size
        self.seq_length = 8
        self.feature_size = self.n_features // self.seq_length
        self.n_rx = n_rx
        self.n_tx = n_tx
        
        # 1D CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        # LSTM处理时序特征
        self.lstm = nn.LSTM(32 * self.feature_size, hidden_size, batch_first=True)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_rx * n_tx)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN特征提取
        x = x.view(batch_size, 1, -1)
        cnn_out = self.cnn(x)
        
        # 重塑以适应LSTM
        lstm_in = cnn_out.view(batch_size, self.seq_length, -1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(lstm_in)
        
        # 全连接层
        x = self.fc(lstm_out[:, -1, :])
        # 重塑输出为 [batch_size, n_rx, n_tx]
        return x.view(batch_size, self.n_rx, self.n_tx) 