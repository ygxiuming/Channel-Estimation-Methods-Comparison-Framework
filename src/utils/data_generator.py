"""
信道数据生成器模块
用于生成MIMO信道数据、导频符号和接收信号
"""

import numpy as np
from typing import Tuple

class ChannelDataGenerator:
    """信道数据生成器类"""
    
    def __init__(self, n_tx: int, n_rx: int, n_pilot: int,
                 channel_type: str = 'rayleigh', rician_k: float = 1.0):
        """
        初始化信道数据生成器
        
        参数:
            n_tx: 发射天线数量
            n_rx: 接收天线数量
            n_pilot: 导频符号长度
            channel_type: 信道类型，'rayleigh' 或 'rician'
            rician_k: Rician K因子，仅在channel_type='rician'时有效
        """
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_pilot = n_pilot
        self.channel_type = channel_type.lower()
        self.rician_k = rician_k
        
        if self.channel_type not in ['rayleigh', 'rician']:
            raise ValueError("信道类型必须是 'rayleigh' 或 'rician'")
        
        if self.n_pilot < self.n_tx:
            raise ValueError("导频长度必须大于等于发射天线数量")
    
    def generate_channel(self, n_samples: int) -> np.ndarray:
        """
        生成MIMO信道矩阵
        
        参数:
            n_samples: 样本数量
        
        返回:
            shape=(n_samples, n_rx, n_tx)的信道矩阵
        """
        # 生成随机复高斯分量
        h_random = (np.random.normal(0, 1/np.sqrt(2), (n_samples, self.n_rx, self.n_tx)) + 
                   1j * np.random.normal(0, 1/np.sqrt(2), (n_samples, self.n_rx, self.n_tx)))
        
        if self.channel_type == 'rayleigh':
            return h_random
        else:  # Rician信道
            # 生成确定性分量（LOS分量）
            h_los = np.ones((n_samples, self.n_rx, self.n_tx)) / np.sqrt(self.n_tx * self.n_rx)
            
            # 组合LOS分量和随机分量
            k = self.rician_k
            h_rician = np.sqrt(k/(k+1)) * h_los + np.sqrt(1/(k+1)) * h_random
            
            return h_rician
    
    def generate_pilot_symbols(self, n_samples: int) -> np.ndarray:
        """
        生成导频符号
        
        参数:
            n_samples: 样本数量
        
        返回:
            shape=(n_samples, n_tx, n_pilot)的导频符号矩阵
        """
        # 使用QPSK调制生成导频
        pilot_real = np.random.choice([-1, 1], size=(n_samples, self.n_tx, self.n_pilot))
        pilot_imag = np.random.choice([-1, 1], size=(n_samples, self.n_tx, self.n_pilot))
        pilot = (pilot_real + 1j * pilot_imag) / np.sqrt(2)
        
        # 归一化功率
        pilot = pilot / np.sqrt(self.n_tx)
        
        return pilot
    
    def generate_received_signal(self, H: np.ndarray, X: np.ndarray, snr_db: float) -> np.ndarray:
        """
        生成接收信号
        
        参数:
            H: shape=(n_samples, n_rx, n_tx)的信道矩阵
            X: shape=(n_samples, n_tx, n_pilot)的发送信号
            snr_db: 信噪比（dB）
        
        返回:
            shape=(n_samples, n_rx, n_pilot)的接收信号
        """
        # 计算信噪比
        snr = 10 ** (snr_db / 10)
        
        # 计算信号功率
        signal_power = np.mean(np.abs(H @ X) ** 2)
        noise_power = signal_power / snr
        
        # 生成噪声
        noise = (np.random.normal(0, np.sqrt(noise_power/2), X.shape[0:1] + (self.n_rx, X.shape[-1])) +
                1j * np.random.normal(0, np.sqrt(noise_power/2), X.shape[0:1] + (self.n_rx, X.shape[-1])))
        
        # 生成接收信号
        Y = np.matmul(H, X) + noise
        
        return Y
    
    def generate(self, n_samples: int, snr_db: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成完整的训练数据集
        
        参数:
            n_samples: 样本数量
            snr_db: 信噪比（dB）
        
        返回:
            (H, X, Y)元组，分别是信道矩阵、导频符号和接收信号
        """
        # 生成信道矩阵
        H = self.generate_channel(n_samples)
        
        # 生成导频符号
        X = self.generate_pilot_symbols(n_samples)
        
        # 生成接收信号
        Y = self.generate_received_signal(H, X, snr_db)
        
        return H, X, Y 