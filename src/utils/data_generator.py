"""
信道数据生成器模块
用于生成MIMO信道数据、导频符号和接收信号

系统模型: Y = H·X + N
    H: (n_samples, n_rx, n_tx)  信道矩阵
    X: (n_samples, n_tx, n_pilot) 导频符号
    N: 复高斯白噪声，每元素方差为 noise_var
"""

import numpy as np
from typing import Tuple

class ChannelDataGenerator:
    """信道数据生成器类"""

    def __init__(self, n_tx: int, n_rx: int, n_pilot: int,
                 channel_type: str = 'rayleigh', rician_k: float = 1.0,
                 pilot_type: str = 'random'):
        """
        初始化信道数据生成器

        参数:
            n_tx: 发射天线数量
            n_rx: 接收天线数量
            n_pilot: 导频符号长度
            channel_type: 信道类型，'rayleigh' 或 'rician'
            rician_k: Rician K因子，仅在channel_type='rician'时有效
            pilot_type: 导频结构，'random'（随机QPSK，历史方式）/
                'dft'（DFT 正交导频）/ 'hadamard'（Walsh-Hadamard 正交导频）。
                正交导频满足 X·Xᴴ = (n_pilot/n_tx)·I，LS 噪声协方差为白化对角阵；
                仍保持每符号功率 1/n_tx，与随机导频口径一致。
        """
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_pilot = n_pilot
        self.channel_type = channel_type.lower()
        self.rician_k = rician_k
        self.pilot_type = pilot_type.lower()

        if self.channel_type not in ['rayleigh', 'rician']:
            raise ValueError("信道类型必须是 'rayleigh' 或 'rician'")
        if self.pilot_type not in ['random', 'dft', 'hadamard']:
            raise ValueError("导频类型必须是 'random' / 'dft' / 'hadamard'")
        if self.pilot_type == 'hadamard' and (self.n_pilot & (self.n_pilot - 1)) != 0:
            raise ValueError("hadamard 导频要求 n_pilot 为 2 的幂")

        if self.n_pilot < self.n_tx:
            raise ValueError("导频长度必须大于等于发射天线数量")

    def generate_channel(self, n_samples: int) -> np.ndarray:
        """
        生成MIMO信道矩阵（E|h_ij|^2 = 1，即各抽头单位平均功率）

        参数:
            n_samples: 样本数量

        返回:
            shape=(n_samples, n_rx, n_tx)的信道矩阵
        """
        # 散射分量: CN(0, 1)
        h_random = (np.random.normal(0, 1/np.sqrt(2), (n_samples, self.n_rx, self.n_tx)) +
                   1j * np.random.normal(0, 1/np.sqrt(2), (n_samples, self.n_rx, self.n_tx)))

        if self.channel_type == 'rayleigh':
            return h_random
        else:  # Rician信道
            # LOS分量: 每抽头单位幅度（|h_los| = 1）
            h_los = np.ones((n_samples, self.n_rx, self.n_tx), dtype=complex)

            # 标准K因子功率分配: E|H|^2 = K/(K+1)*|h_los|^2 + 1/(K+1)*E|h_scatter|^2 = 1
            k = self.rician_k
            h_rician = np.sqrt(k/(k+1)) * h_los + np.sqrt(1/(k+1)) * h_random

            return h_rician

    def _orthogonal_pilot_matrix(self, n_samples: int) -> np.ndarray:
        """构造确定性的正交导频：X·Xᴴ = (n_pilot/n_tx)·I，符号功率 1/n_tx"""
        t = np.arange(self.n_tx)[:, None]
        p = np.arange(self.n_pilot)[None, :]
        if self.pilot_type == 'dft':
            entry = np.exp(2j * np.pi * t * p / self.n_pilot)
        else:  # hadamard：Walsh-Hadamard 矩阵前 n_tx 行（符号内相位由 +1/-1 承载）
            entry = self._walsh_hadamard(self.n_pilot)[:self.n_tx, :].astype(complex)
        return np.broadcast_to(entry, (n_samples, self.n_tx, self.n_pilot)).copy() / np.sqrt(self.n_tx)

    @staticmethod
    def _walsh_hadamard(n: int) -> np.ndarray:
        """递归构造 n×n 的 Walsh-Hadamard 矩阵（n 为 2 的幂），元素 ±1"""
        H = np.ones((1, 1), dtype=float)
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        return H

    def generate_pilot_symbols(self, n_samples: int) -> np.ndarray:
        """
        生成导频符号（QPSK，总发射功率归一化）

        参数:
            n_samples: 样本数量

        返回:
            shape=(n_samples, n_tx, n_pilot)的导频符号矩阵
        """
        if self.pilot_type in ('dft', 'hadamard'):
            return self._orthogonal_pilot_matrix(n_samples)

        # 使用QPSK调制生成导频（单符号单位功率）
        pilot_real = np.random.choice([-1, 1], size=(n_samples, self.n_tx, self.n_pilot))
        pilot_imag = np.random.choice([-1, 1], size=(n_samples, self.n_tx, self.n_pilot))
        pilot = (pilot_real + 1j * pilot_imag) / np.sqrt(2)

        # 总发射功率归一化（跨天线）
        pilot = pilot / np.sqrt(self.n_tx)

        return pilot

    def generate_received_signal(self, H: np.ndarray, X: np.ndarray,
                                 snr_db: float) -> Tuple[np.ndarray, float]:
        """
        生成接收信号

        参数:
            H: shape=(n_samples, n_rx, n_tx)的信道矩阵
            X: shape=(n_samples, n_tx, n_pilot)的发送信号
            snr_db: 信噪比（dB），定义为接收端平均信号功率与噪声功率之比

        返回:
            (Y, noise_var):
                Y: shape=(n_samples, n_rx, n_pilot)的接收信号
                noise_var: 噪声每维（复数）方差，供LMMSE和BER仿真使用
        """
        # 计算信噪比
        snr = 10 ** (snr_db / 10)

        # 计算平均信号功率（每个接收元素）
        signal_power = np.mean(np.abs(H @ X) ** 2)
        noise_power = signal_power / snr

        # 生成噪声
        noise = (np.random.normal(0, np.sqrt(noise_power/2), X.shape[0:1] + (self.n_rx, X.shape[-1])) +
                1j * np.random.normal(0, np.sqrt(noise_power/2), X.shape[0:1] + (self.n_rx, X.shape[-1])))

        # 生成接收信号
        Y = np.matmul(H, X) + noise

        return Y, float(noise_power)

    def generate(self, n_samples: int, snr_db: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        生成完整的数据集

        参数:
            n_samples: 样本数量
            snr_db: 信噪比（dB）

        返回:
            (H, X, Y, noise_var)元组，分别是信道矩阵、导频符号、接收信号和噪声方差
        """
        # 生成信道矩阵
        H = self.generate_channel(n_samples)

        # 生成导频符号
        X = self.generate_pilot_symbols(n_samples)

        # 生成接收信号
        Y, noise_var = self.generate_received_signal(H, X, snr_db)

        return H, X, Y, noise_var


class OFDMDataGenerator:
    """
    MIMO-OFDM 频率选择性信道数据生成器

    系统模型（单导频 OFDM 符号快照，梳状导频 comb pilots）：
        H: (n_samples, n_rx, n_pilot, n_tx)  频域信道网格
           H[rx, k, tx] = Σ_l h[rx, tx, l]·exp(-2πj·k·l/N)，N = n_pilot（子载波数）
           时域抽头 h ~ CN(0, pdp_l) 独立，pdp 为指数功率时延谱并归一化
           （Σpdp = 1，因此每个频域信道元素平均功率为 1，NMSE 口径与平坦信道一致）
        X: (n_samples, n_tx, n_pilot) 梳状导频——发射天线 t 仅在子载波
           k ≡ t (mod n_tx) 上发送单位模（功率 1/n_tx）的 QPSK 符号，其余为 0
        Y: (n_samples, n_rx, n_pilot) = Σ_tx H[rx, k, tx]·X[tx, k] + N

    估计任务本质：利用子载波间的频率相关性，把稀疏的梳状观测插值到
    整个频域网格——传统方法只能线性插值，深度学习可学习非线性插值。
    """

    def __init__(self, n_tx: int, n_rx: int, n_sc: int, n_taps: int = 4,
                 pdp_decay: float = 0.35):
        if n_sc < n_tx or n_tx < 1:
            raise ValueError("OFDM 要求子载波数 n_sc ≥ n_tx")
        if n_sc % n_tx != 0:
            raise ValueError("梳状导频要求子载波数 n_sc 是 n_tx 的整数倍")
        if n_taps < 1 or n_taps > n_sc:
            raise ValueError("抽头数要求 1 ≤ n_taps ≤ n_sc")

        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_pilot = n_sc          # 复用 n_pilot 字段表示子载波数，下游代码共用
        self.n_taps = n_taps
        self.pdp_decay = pdp_decay
        # 归一化指数功率时延谱 pdp_l ∝ exp(-decay·l), Σ pdp = 1
        raw = np.exp(-pdp_decay * np.arange(n_taps))
        self.pdp = raw / raw.sum()

        # DFT 展开矩阵：freq_gain[k, l] = exp(-2πj·k·l/N)，频域信道 =
        # 时域抽头 h 的 N 点 DFT 展开（生成模型的物理内核）
        self.freq_gain = np.exp(-2j * np.pi *
                                np.outer(np.arange(n_sc), np.arange(n_taps)) / n_sc)

    def generate_channel(self, n_samples: int) -> np.ndarray:
        """生成频域信道网格 (n, n_rx, n_sc, n_tx)：E|H|² = Σpdp = 1"""
        std = np.sqrt(self.pdp / 2)[None, None, None, :]
        taps = (np.random.normal(0, std, (n_samples, self.n_rx, self.n_tx, self.n_taps)) +
                1j * np.random.normal(0, std, (n_samples, self.n_rx, self.n_tx, self.n_taps)))
        # H[n, r, k, t] = Σ_l taps[n, r, t, l]·freq_gain[k, l]
        H = np.einsum('nrtl,kl->nrkt', taps, self.freq_gain)
        return H

    def generate_pilot_symbols(self, n_samples: int) -> np.ndarray:
        """梳状导频 (n, n_tx, n_sc)：tx t 仅在子载波 k ≡ t (mod n_tx) 非零"""
        X = np.zeros((n_samples, self.n_tx, self.n_pilot), dtype=complex)
        # 符号值：单位模 QPSK，功率 1/n_tx（与平坦信道导频口径一致）
        for t in range(self.n_tx):
            cols = np.arange(t, self.n_pilot, self.n_tx)
            X[:, t, cols] = (np.random.choice([-1, 1], (n_samples, cols.size)) +
                             1j * np.random.choice([-1, 1], (n_samples, cols.size))) \
                / (np.sqrt(2) * np.sqrt(self.n_tx))
        return X

    def generate_received_signal(self, H: np.ndarray, X: np.ndarray,
                                 snr_db: float):
        """
        生成接收信号 Y = Σ_tx H[:, k, tx]·X[tx, k] + N

        返回 (Y, noise_var)：Y (n, n_rx, n_sc)，noise_var 为每复元素方差，
        SNR 口径与平坦信道一致（全符号平均信号功率 / 噪声功率）。
        """
        snr = 10 ** (snr_db / 10)
        # Y[n, rx, k] = Σ_tx H[n, rx, k, tx]·X[n, tx, k]
        Y_signal = np.einsum('nrkt,ntk->nrk', H, X)
        signal_power = float(np.mean(np.abs(Y_signal) ** 2))
        noise_power = signal_power / snr
        std = np.sqrt(noise_power / 2)
        Y = (Y_signal + np.random.normal(0, std, Y_signal.shape) +
             1j * np.random.normal(0, std, Y_signal.shape))
        return Y, noise_power

    def generate(self, n_samples: int, snr_db: float):
        """生成完整数据集 (H, X, Y, noise_var)，H 为频域 4 维网格"""
        H = self.generate_channel(n_samples)
        X = self.generate_pilot_symbols(n_samples)
        Y, noise_var = self.generate_received_signal(H, X, snr_db)
        return H, X, Y, noise_var
