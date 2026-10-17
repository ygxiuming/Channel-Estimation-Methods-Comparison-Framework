"""
深度学习信道估计模型模块

所有模型统一约定：
- 输入: 展平的实数特征向量 (batch, input_size)，由 utils/features.py 按输入模式构造：
    * 'raw'         : [Re(Y), Im(Y), Re(X), Im(X)] 展平（逐导频点）
    * 'raw_sym'     : 同 raw 内容，按导频符号分组排布（保持时序模型局部性）
    * 'ls_refine'   : [Re(H_ls), Im(H_ls)] 展平
    * 'ls_refine_x' : ls_refine + 导频实虚部（供 LmmseNet 物理层使用）
- 输出: (batch, n_rx, 2·n_tx)，最后一维为 [实部 | 虚部] 拼接，
        由调用方重组为复数信道后再计算指标（与传统方法同口径）

包含常规模型（CNN/RNN/LSTM/GRU/Hybrid）与模型驱动 unfolding 模型 LmmseNet。
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


class LmmseNetEstimator(BaseEstimator):
    """
    模型驱动 Deep Unfolding 估计器（LMMSE-Net 风格）

    结构 = 内嵌物理层 + 可学习校正：

    1. 物理层在前向中显式执行逐样本 LMMSE 滤波（复数批量求解，无可学习矩阵）：

       A_i = R·I + r·(X·Xᴴ)⁻¹        （逐样本）
       H_phys = R·solve(A_i, R·H_lsᵀ)ᵀ

       其中 R = exp(log_gain)（信道行功率尺度）、r = exp(log_noise)
       （等效噪声方差）为两个可学习的全局标量。
       R=1、r=噪声方差时物理层严格退化为最优 LMMSE。
    2. 校正头（MLP）以 [H_ls, H_phys] 为输入，输出残差修正 Δ，
       最终输出 = H_phys + Δ。校正头末层零初始化——
       训练从"纯 LMMSE"起步，网络只学习物理层未覆盖的偏差。

    输入必须为 'ls_refine_x' 模式特征（含 H_ls 与 X 实虚部），
    否则物理层无法获得逐样本 (X·Xᴴ)⁻¹，构造时即报错。
    """

    def __init__(self, input_size: int, n_rx: int, n_tx: int, n_pilot: int,
                 dropout: float = 0.3, init_noise: float = 0.01):
        super().__init__(input_size, n_rx, n_tx, dropout)
        self.n_pilot = n_pilot
        self.h_size = n_rx * n_tx
        self.x_size = n_tx * n_pilot
        expected = 2 * self.h_size + 2 * self.x_size
        if input_size != expected:
            raise ValueError(
                f"LmmseNet 需要 'ls_refine_x' 模式的输入特征："
                f"期望输入维度 {expected}（H_ls 实虚部 + X 实虚部），"
                f"实际 {input_size}。请检查 dl.input_modes 是否包含 ls_refine_x")

        self.log_gain = nn.Parameter(torch.zeros(1))          # R = exp(log_gain)
        self.log_noise = nn.Parameter(torch.log(
            torch.as_tensor(float(init_noise))))              # r = exp(log_noise)

        hidden = max(512, self.output_size)
        self.head = nn.Sequential(
            nn.Linear(4 * self.h_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.output_size),
        )
        # 校正头末层零初始化：起步即纯物理层（Y = LMMSE），只学残差
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _physics_layer(self, H_ls: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """逐样本 LMMSE 物理层（复数批量求解）

        参数:
            H_ls: (B, n_rx, n_tx) 复数 LS 估计
            X:    (B, n_tx, n_pilot) 复数导频
        返回:
            H_phys: (B, n_rx, n_tx) 复数物理层输出
        """
        G = torch.matmul(X, X.conj().transpose(-1, -2))       # XXᴴ (B, n_tx, n_tx)
        eye = torch.eye(self.n_tx, dtype=G.dtype, device=G.device)
        gain = self.log_gain.exp()
        noise = self.log_noise.exp()
        A = gain * eye + noise * torch.linalg.inv(G + 1e-10 * eye)   # R + rC
        # ĥ_col = A⁻¹·R·h_col，对全部接收天线行批量求解
        rhs = torch.linalg.solve(A, gain * H_ls.transpose(-1, -2))
        h_phys = rhs.transpose(-1, -2)                        # (B, n_rx, n_tx)
        return h_phys

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        # 解包特征：[H_ls 实 | H_ls 虚 | X 实 | X 虚]（与 utils.features.ls_refine_x 一致）
        h_re = x[:, :self.h_size].view(batch, self.n_rx, self.n_tx)
        h_im = x[:, self.h_size:2 * self.h_size].view(batch, self.n_rx, self.n_tx)
        x_re = x[:, 2 * self.h_size:2 * self.h_size + self.x_size] \
            .view(batch, self.n_tx, self.n_pilot)
        x_im = x[:, 2 * self.h_size + self.x_size:] \
            .view(batch, self.n_tx, self.n_pilot)

        H_ls = torch.complex(h_re, h_im)
        X = torch.complex(x_re, x_im)
        H_phys = self._physics_layer(H_ls, X)

        feats = torch.cat([H_phys.real.flatten(1), H_phys.imag.flatten(1),
                           H_ls.real.flatten(1), H_ls.imag.flatten(1)], dim=1)
        delta = self.head(feats).view(batch, self.n_rx, 2 * self.n_tx)

        out = torch.empty_like(delta)
        out[..., :self.n_tx] = H_phys.real + delta[..., :self.n_tx]
        out[..., self.n_tx:] = H_phys.imag + delta[..., self.n_tx:]
        return out


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
