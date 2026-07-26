"""
信道数据预处理模块

设计原则：
1. 归一化使用全局统一标量缩放，H/Y 共用同一因子，
   严格保持 Y = H·X + N 的线性关系；
2. 不提供逐元素"异常值替换"——瑞利衰落信道的重尾是物理特性，
   逐元素裁剪会逐样本破坏信道模型与接收信号的对应关系；
3. 数据增强以 (H, X, Y) 三元组为单位同步变换，保证物理一致性。
"""

import numpy as np
from typing import Optional, List, Tuple


class ChannelPreprocessor:
    """
    信道数据预处理器（全局功率缩放）

    对复数数据做单一标量缩放: data / scale。
    H 与 Y 使用同一 scale 时，线性关系 Y = H·X + N 精确保持。
    """

    def __init__(self, normalization: str = 'power'):
        """
        参数:
            normalization: 'power'（按RMS功率缩放）或 'none'（不缩放）
        """
        self.normalization = normalization.lower()
        if self.normalization not in ['power', 'none']:
            raise ValueError("归一化方法必须是 'power' 或 'none'")
        self.scale: Optional[float] = None

    def fit(self, data: np.ndarray) -> None:
        """根据参考数据计算缩放因子（复数RMS）"""
        if self.normalization == 'power':
            rms = float(np.sqrt(np.mean(np.abs(data) ** 2)))
            self.scale = rms if rms > 0 else 1.0
        else:
            self.scale = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        """应用缩放"""
        if self.scale is None:
            raise ValueError("请先调用fit方法计算缩放参数")
        return data / self.scale

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """计算缩放参数并应用"""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反缩放，恢复原始量纲"""
        if self.scale is None:
            raise ValueError("请先调用fit方法计算缩放参数")
        return data * self.scale


def augment_triplets(H: np.ndarray, X: np.ndarray, Y: np.ndarray,
                     methods: Optional[List[str]] = None,
                     phase_shift_range: Tuple[float, float] = (-np.pi, np.pi),
                     magnitude_scale_range: Tuple[float, float] = (0.8, 1.25)
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    物理一致的数据增强：对 (H, X, Y) 三元组做同步变换。

    所有变换均为逐样本的公共标量变换，保证变换后仍满足 Y' = H'·X + N'
    （N' 与 N 统计同分布）：

    - 'phase_shift':     H -> H·e^{jφ},  Y -> Y·e^{jφ}  （逐样本公共相位 φ）
    - 'magnitude_scale': H -> H·s,       Y -> Y·s       （逐样本公共实数增益 s）

    注意：逐元素独立的相位/幅度扰动对MIMO相干信道没有物理意义，已移除；
    对 H 单独加噪而不更新 Y 会破坏真值对应关系，同样已移除。

    参数:
        H: 信道矩阵 (n_samples, n_rx, n_tx)
        X: 导频符号 (n_samples, n_tx, n_pilot)
        Y: 接收信号 (n_samples, n_rx, n_pilot)
        methods: 增强方法列表，可选 ['phase_shift', 'magnitude_scale']
        phase_shift_range: 相位偏移范围（弧度）
        magnitude_scale_range: 幅度缩放范围

    返回:
        (H_aug, X, Y_aug) 增强后的三元组（X不变）
    """
    if methods is None:
        methods = ['phase_shift', 'magnitude_scale']

    H_aug = H.copy()
    Y_aug = Y.copy()
    n_samples = H.shape[0]
    sample_shape = (n_samples,) + (1,) * (H.ndim - 1)

    for method in methods:
        if method == 'phase_shift':
            phase = np.random.uniform(phase_shift_range[0],
                                      phase_shift_range[1],
                                      sample_shape)
            rot = np.exp(1j * phase)
            H_aug = H_aug * rot
            Y_aug = Y_aug * rot
        elif method == 'magnitude_scale':
            scale = np.random.uniform(magnitude_scale_range[0],
                                      magnitude_scale_range[1],
                                      sample_shape)
            H_aug = H_aug * scale
            Y_aug = Y_aug * scale
        else:
            raise ValueError(f"未知的增强方法: {method}，"
                             f"可选 ['phase_shift', 'magnitude_scale']")

    return H_aug, X, Y_aug
