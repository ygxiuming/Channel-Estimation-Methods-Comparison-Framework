"""
信道数据预处理模块
用于数据归一化和异常值处理
"""

import numpy as np
from typing import Optional, Union, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class ChannelPreprocessor:
    """信道数据预处理器"""
    
    def __init__(self, normalization: str = 'z-score',
                 remove_outliers: bool = False,
                 outlier_threshold: float = 3.0):
        """
        初始化预处理器
        
        参数:
            normalization: 归一化方法，'z-score' 或 'min-max'
            remove_outliers: 是否移除异常值
            outlier_threshold: 异常值阈值（标准差的倍数）
        """
        self.normalization = normalization.lower()
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        
        if self.normalization not in ['z-score', 'min-max']:
            raise ValueError("归一化方法必须是 'z-score' 或 'min-max'")
        
        # 存储归一化参数
        self.mean_real = None
        self.std_real = None
        self.mean_imag = None
        self.std_imag = None
        self.min_real = None
        self.max_real = None
        self.min_imag = None
        self.max_imag = None
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        移除异常值
        
        参数:
            data: 输入数据
        
        返回:
            处理后的数据
        """
        if not self.remove_outliers:
            return data
        
        # 分别处理实部和虚部
        real_part = np.real(data)
        imag_part = np.imag(data)
        
        # 计算均值和标准差
        mean_real = np.mean(real_part)
        std_real = np.std(real_part)
        mean_imag = np.mean(imag_part)
        std_imag = np.std(imag_part)
        
        # 创建掩码
        real_mask = np.abs(real_part - mean_real) <= self.outlier_threshold * std_real
        imag_mask = np.abs(imag_part - mean_imag) <= self.outlier_threshold * std_imag
        mask = real_mask & imag_mask
        
        # 将异常值替换为均值
        real_part[~real_mask] = mean_real
        imag_part[~imag_mask] = mean_imag
        
        return real_part + 1j * imag_part
    
    def fit(self, data: np.ndarray) -> None:
        """
        计算归一化参数
        
        参数:
            data: 输入数据
        """
        # 首先移除异常值
        data = self._remove_outliers(data)
        
        # 分离实部和虚部
        real_part = np.real(data)
        imag_part = np.imag(data)
        
        if self.normalization == 'z-score':
            self.mean_real = np.mean(real_part)
            self.std_real = np.std(real_part)
            self.mean_imag = np.mean(imag_part)
            self.std_imag = np.std(imag_part)
        else:  # min-max归一化
            self.min_real = np.min(real_part)
            self.max_real = np.max(real_part)
            self.min_imag = np.min(imag_part)
            self.max_imag = np.max(imag_part)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        应用归一化
        
        参数:
            data: 输入数据
        
        返回:
            归一化后的数据
        """
        # 首先移除异常值
        data = self._remove_outliers(data)
        
        # 分离实部和虚部
        real_part = np.real(data)
        imag_part = np.imag(data)
        
        if self.normalization == 'z-score':
            if self.mean_real is None or self.std_real is None:
                raise ValueError("请先调用fit方法计算归一化参数")
            
            real_norm = (real_part - self.mean_real) / self.std_real
            imag_norm = (imag_part - self.mean_imag) / self.std_imag
        else:  # min-max归一化
            if self.min_real is None or self.max_real is None:
                raise ValueError("请先调用fit方法计算归一化参数")
            
            real_norm = (real_part - self.min_real) / (self.max_real - self.min_real)
            imag_norm = (imag_part - self.min_imag) / (self.max_imag - self.min_imag)
        
        return real_norm + 1j * imag_norm
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        计算归一化参数并应用归一化
        
        参数:
            data: 输入数据
        
        返回:
            归一化后的数据
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        反归一化
        
        参数:
            data: 归一化后的数据
        
        返回:
            原始尺度的数据
        """
        real_part = np.real(data)
        imag_part = np.imag(data)
        
        if self.normalization == 'z-score':
            if self.mean_real is None or self.std_real is None:
                raise ValueError("请先调用fit方法计算归一化参数")
            
            real_orig = real_part * self.std_real + self.mean_real
            imag_orig = imag_part * self.std_imag + self.mean_imag
        else:  # min-max归一化
            if self.min_real is None or self.max_real is None:
                raise ValueError("请先调用fit方法计算归一化参数")
            
            real_orig = real_part * (self.max_real - self.min_real) + self.min_real
            imag_orig = imag_part * (self.max_imag - self.min_imag) + self.min_imag
        
        return real_orig + 1j * imag_orig

def augment_data(data: np.ndarray,
                 methods: Optional[List[str]] = None,
                 noise_std: float = 0.01,
                 phase_shift_range: Tuple[float, float] = (-0.1, 0.1),
                 magnitude_scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """
    数据增强
    
    参数:
        data: 输入数据
        methods: 增强方法列表，可选['noise', 'phase_shift', 'magnitude_scale']
        noise_std: 高斯噪声标准差
        phase_shift_range: 相位偏移范围（弧度）
        magnitude_scale_range: 幅度缩放范围
    
    返回:
        增强后的数据
    """
    if methods is None:
        methods = ['noise', 'phase_shift', 'magnitude_scale']
    
    augmented_data = data.copy()
    
    for method in methods:
        if method == 'noise':
            # 添加复高斯噪声
            noise = (np.random.normal(0, noise_std, data.shape) +
                    1j * np.random.normal(0, noise_std, data.shape))
            augmented_data += noise
        
        elif method == 'phase_shift':
            # 随机相位偏移
            phase_shift = np.random.uniform(phase_shift_range[0],
                                         phase_shift_range[1],
                                         data.shape)
            augmented_data *= np.exp(1j * phase_shift)
        
        elif method == 'magnitude_scale':
            # 随机幅度缩放
            scale = np.random.uniform(magnitude_scale_range[0],
                                    magnitude_scale_range[1],
                                    data.shape)
            augmented_data *= scale
    
    return augmented_data 