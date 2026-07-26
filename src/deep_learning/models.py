"""
深度学习信道估计模型模块

所有模型统一约定：
- 输入: 展平的实数特征向量 (batch, input_size)，由调用方根据输入模式构造：
    * 'raw'       : [Re(Y), Im(Y), Re(X), Im(X)] 展平，input_size = 2·n_pilot·(n_rx+n_tx)
    * 'ls_refine' : [Re(H_ls), Im(H_ls)] 展平，input_size = 2·n_rx·n_tx
- 输出: (batch, n_rx, 2·n_tx)，最后一维为 [实部 | 虚部] 拼接，
        由调用方重组为复数信道后再计算指标（与传统方法同口径）

所有维度（n_rx, n_tx, input_size）由构造函数显式传入，不做任何隐式假设。
"""

import torch
import torch.nn as nn


def _split_sequence(n_features: int, max_seq_len: int = 16):
    """
    将特征维拆分为 (seq_length, feature_size)，要求严格整除。

    选取不超过 max_seq_len 的 n_features 的最大因子作为 seq_length，
    避免原实现中硬编码 seq_length=8 导致的整除崩溃。
    """
    for seq_len in range(min(max_seq_len, n_features), 0, -1):
        if n_features % seq_len == 0:
            return seq_len, n_features // seq_len
    return 1, n_features  # n_features >= 1 时必然可达


def _make_fc(n_in: int, n_out: int, dropout: float = 0.3) -> nn.Sequential:
    """两层全连接输出头"""
    hidden = max(512, n_out)
    return nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_out)
    )


class BaseEstimator(nn.Module):
    """基类：统一处理输出维度 (n_rx, 2·n_tx)"""

    def __init__(self, input_size: int, n_rx: int, n_tx: int, dropout: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.n_rx = n_rx
        self.n_tx = n_tx
        self.output_size = n_rx * 2 * n_tx
        self.dropout = dropout

    def _reshape_output(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), self.n_rx, 2 * self.n_tx)


class CNNEstimator(BaseEstimator):
    """基于1D CNN的信道估计器"""

    def __init__(self, input_size: int, n_rx: int, n_tx: int, dropout: float = 0.3):
        super().__init__(input_size, n_rx, n_tx, dropout)

        self.cnn = nn.Sequential(
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

        self.fc = _make_fc(self.input_size, self.output_size, dropout)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return self._reshape_output(x)


class RNNEstimator(BaseEstimator):
    """基于RNN的信道估计器"""

    def __init__(self, input_size: int, n_rx: int, n_tx: int,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__(input_size, n_rx, n_tx, dropout)

        self.seq_length, self.feature_size = _split_sequence(self.input_size)

        self.rnn = nn.RNN(self.feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = _make_fc(hidden_size, self.output_size, dropout)

    def forward(self, x):
        x = x.view(x.size(0), self.seq_length, self.feature_size)
        out, _ = self.rnn(x)
        x = self.fc(out[:, -1, :])
        return self._reshape_output(x)


class LSTMEstimator(BaseEstimator):
    """基于LSTM的信道估计器"""

    def __init__(self, input_size: int, n_rx: int, n_tx: int,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__(input_size, n_rx, n_tx, dropout)

        self.seq_length, self.feature_size = _split_sequence(self.input_size)

        self.lstm = nn.LSTM(self.feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = _make_fc(hidden_size, self.output_size, dropout)

    def forward(self, x):
        x = x.view(x.size(0), self.seq_length, self.feature_size)
        out, _ = self.lstm(x)
        x = self.fc(out[:, -1, :])
        return self._reshape_output(x)


class GRUEstimator(BaseEstimator):
    """基于GRU的信道估计器"""

    def __init__(self, input_size: int, n_rx: int, n_tx: int,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__(input_size, n_rx, n_tx, dropout)

        self.seq_length, self.feature_size = _split_sequence(self.input_size)

        self.gru = nn.GRU(self.feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = _make_fc(hidden_size, self.output_size, dropout)

    def forward(self, x):
        x = x.view(x.size(0), self.seq_length, self.feature_size)
        out, _ = self.gru(x)
        x = self.fc(out[:, -1, :])
        return self._reshape_output(x)


class HybridEstimator(BaseEstimator):
    """混合深度学习估计器（1D CNN + LSTM）"""

    def __init__(self, input_size: int, n_rx: int, n_tx: int,
                 hidden_size: int = 256, dropout: float = 0.3):
        super().__init__(input_size, n_rx, n_tx, dropout)

        self.seq_length, self.feature_size = _split_sequence(self.input_size)

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

        self.fc = _make_fc(hidden_size, self.output_size, dropout)

    def forward(self, x):
        batch_size = x.size(0)

        # CNN特征提取
        x = x.view(batch_size, 1, -1)
        cnn_out = self.cnn(x)

        # 重塑为 (batch, seq_length, 32*feature_size) 以适应LSTM
        lstm_in = cnn_out.view(batch_size, self.seq_length, -1)

        # LSTM处理
        lstm_out, _ = self.lstm(lstm_in)

        x = self.fc(lstm_out[:, -1, :])
        return self._reshape_output(x)
