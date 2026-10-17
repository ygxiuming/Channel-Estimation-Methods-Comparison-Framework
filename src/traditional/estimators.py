"""
传统信道估计方法模块
包含LS、LMMSE和ML估计器，以及统一的复数域性能指标计算

系统模型: Y = H·X + N
    Y: (n_samples, n_rx, n_pilot)  接收信号
    X: (n_samples, n_tx, n_pilot)  导频符号
    H: (n_samples, n_rx, n_tx)     信道矩阵
    N: 复高斯白噪声，每元素方差 noise_var
"""

import numpy as np
from typing import Dict, Optional


class LSEstimator:
    """最小二乘(LS)估计器: H_ls = Y·Xᴴ·(X·Xᴴ)⁻¹"""

    def __init__(self, regularization: bool = False, lambda_: float = 0.01):
        """
        参数:
            regularization: 是否使用对角加载正则化
            lambda_: 正则化参数
        """
        self.regularization = regularization
        self.lambda_ = lambda_

    def estimate(self, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        执行LS估计（批量向量化）

        参数:
            Y: 接收信号，shape=(n_samples, n_rx, n_pilot)
            X: 导频符号，shape=(n_samples, n_tx, n_pilot)

        返回:
            H_est: 估计的信道矩阵，shape=(n_samples, n_rx, n_tx)
        """
        X_H = np.conjugate(np.transpose(X, (0, 2, 1)))  # (n, n_pilot, n_tx)
        XX_H = np.matmul(X, X_H)                        # (n, n_tx, n_tx)

        if self.regularization:
            n_tx = X.shape[1]
            XX_H = XX_H + self.lambda_ * np.eye(n_tx)[np.newaxis, :, :]

        # H = Y·Xᴴ·(XXᴴ)⁻¹ = ( (XXᴴ)⁻¹·X·Yᴴ )ᴴ  （XXᴴ为Hermitian）
        # 用批量 solve 避免显式求逆，提高数值稳定性
        Y_H = np.conjugate(np.transpose(Y, (0, 2, 1)))  # (n, n_pilot, n_rx)
        M = np.matmul(X, Y_H)                           # (n, n_tx, n_rx)
        try:
            S = np.linalg.solve(XX_H, M)                # (n, n_tx, n_rx)
        except np.linalg.LinAlgError:
            # 批量求解失败时回退到逐样本伪逆
            S = np.stack([np.linalg.pinv(XX_H[i]) @ M[i]
                          for i in range(Y.shape[0])], axis=0)

        H_est = np.conjugate(np.transpose(S, (0, 2, 1)))  # (n, n_rx, n_tx)
        return H_est


class LMMSEEstimator:
    """
    线性最小均方误差(LMMSE)估计器

    基于LS估计 h_ls = h + ñ，其中 ñ 的协方差（每个接收天线行）为
    noise_var·(X·Xᴴ)⁻¹。逐接收天线行的LMMSE滤波：

        ĥ = R_h · (R_h + noise_var·(X·Xᴴ)⁻¹)⁻¹ · h_ls

    其中 R_h 为单行信道向量（长度 n_tx）的相关矩阵。
    各接收天线行独立同分布时，行滤波即为全局最优线性估计。
    """

    def __init__(self, noise_var: float,
                 correlation_method: str = 'theoretical',
                 channel_cov: Optional[np.ndarray] = None):
        """
        参数:
            noise_var: 噪声方差（复数每元素），应来自数据生成器
            correlation_method: 信道相关矩阵估计方法，
                'theoretical' —— 独立衰落信道 R_h = E|h|²·I（无数据泄漏，默认）
                'sample'      —— 使用外部提供的样本相关矩阵 channel_cov
            channel_cov: 样本相关矩阵 (n_tx, n_tx)，
                仅 correlation_method='sample' 时需要；
                注意：必须只在训练集上估计，避免测试集数据泄漏
        """
        self.noise_var = float(noise_var)
        self.correlation_method = correlation_method
        self.channel_cov = channel_cov

        if correlation_method not in ['sample', 'theoretical']:
            raise ValueError("correlation_method必须是'sample'或'theoretical'")
        if correlation_method == 'sample' and channel_cov is None:
            raise ValueError("correlation_method='sample'时必须提供channel_cov")

    def estimate(self, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        执行LMMSE估计（批量向量化）

        参数:
            Y: 接收信号，shape=(n_samples, n_rx, n_pilot)
            X: 导频符号，shape=(n_samples, n_tx, n_pilot)

        返回:
            H_est: 估计的信道矩阵，shape=(n_samples, n_rx, n_tx)
        """
        n_samples, n_rx, _ = Y.shape
        n_tx = X.shape[1]

        # LS估计（无正则化，LMMSE本身提供正则）
        H_ls = LSEstimator(regularization=False).estimate(Y, X)

        # 信道相关矩阵 R_h (n_tx, n_tx)
        if self.correlation_method == 'theoretical':
            R_h = np.eye(n_tx, dtype=complex)  # E|h|² = 1 的独立衰落
        else:
            R_h = self.channel_cov + 1e-10 * np.eye(n_tx)  # 数值稳定

        # 每个样本的 LS 噪声协方差: noise_var·(X·Xᴴ)⁻¹
        X_H = np.conjugate(np.transpose(X, (0, 2, 1)))
        XX_H = np.matmul(X, X_H)                            # (n, n_tx, n_tx)
        XX_H_inv = np.linalg.inv(XX_H)                      # (n, n_tx, n_tx)

        # A = R_h + noise_var·(XXᴴ)⁻¹
        A = R_h[np.newaxis, :, :] + self.noise_var * XX_H_inv  # (n, n_tx, n_tx)

        # 逐样本滤波矩阵 M = R_h·A⁻¹，则列向量形式 ĥ = M·h_ls。
        # 注意：信道行向量转为列向量是普通转置（不带共轭），
        # 因此批量行向量形式为 H_est = H_ls @ M.T（普通转置）。
        # 用普通转置的批量solve避免显式求逆：
        # M = R_h·A⁻¹ = ( A⁻ᵀ·R_hᵀ )ᵀ = solve(Aᵀ, R_hᵀ)ᵀ
        A_T = np.transpose(A, (0, 2, 1))
        R_h_T = np.broadcast_to(np.transpose(R_h), (n_samples, n_tx, n_tx))
        M = np.transpose(np.linalg.solve(A_T, R_h_T), (0, 2, 1))
        H_est = np.matmul(H_ls, np.transpose(M, (0, 2, 1)))

        return H_est


class MLEstimator:
    """
    最大似然(ML)估计器

    在线性高斯模型 Y = H·X + N、X 已知的条件下，ML解具有闭式形式，
    与无正则化的LS解严格等价：H_ml = Y·Xᴴ·(X·Xᴴ)⁻¹。
    因此本实现直接使用闭式解（原迭代实现在数学上冗余且收敛行为依赖
    人为的梯度缩放常数，已移除）。
    """

    def __init__(self):
        pass

    def estimate(self, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        执行ML估计（闭式解，等价于无正则LS）

        参数:
            Y: 接收信号，shape=(n_samples, n_rx, n_pilot)
            X: 导频符号，shape=(n_samples, n_tx, n_pilot)

        返回:
            H_est: 估计的信道矩阵，shape=(n_samples, n_rx, n_tx)
        """
        return LSEstimator(regularization=False).estimate(Y, X)


def estimate_noise_var_blind(Y: np.ndarray, X: np.ndarray,
                             H_ls: Optional[np.ndarray] = None) -> float:
    """
    盲噪声方差估计（基于 LS 残差，不使用数据生成器提供的"上帝"数值）

    原理：LS 估计的残差 R = Y - H_ls·X 严格落在导频列空间的正交补中。
    对每个接收天线行，该正交补维度为 (n_pilot - n_tx)，且残差每个元素
    与信道输出噪声同方差（每复元素方差 noise_var），因此无偏估计为：

        noise_var = Σ|R|² / (n_samples · n_rx · (n_pilot - n_tx))

    仅使用观测 (Y, X)，为 LMMSE 提供"不知道上帝数值"的操作点，
    使传统方法与深度学习方法在公平性上对齐（DL 同样不知道真实噪声方差）。

    参数:
        Y: 接收信号，shape=(n_samples, n_rx, n_pilot)
        X: 导频符号，shape=(n_samples, n_tx, n_pilot)
        H_ls: 可选，已有的 LS 估计（复数）；缺省时内部现算

    返回:
        噪声方差（每复数元素）的浮点估计值
    """
    n_tx, n_pilot = X.shape[1], X.shape[-1]
    if n_pilot <= n_tx:
        raise ValueError(
            "盲噪声方差估计要求导频长度大于发射天线数 (n_pilot > n_tx)，"
            "否则 LS 残差自由度为零")

    if H_ls is None:
        H_ls = LSEstimator(regularization=False).estimate(Y, X)

    residual = Y - H_ls @ X                            # (n, n_rx, n_pilot)
    # E[Σ|r|² / 元素数] = noise_var · (n_pilot - n_tx) / n_pilot
    return float(np.mean(np.abs(residual) ** 2) * n_pilot / (n_pilot - n_tx))


def calculate_performance_metrics(H_true: np.ndarray, H_est: np.ndarray) -> Dict[str, float]:
    """
    计算性能指标（统一在复数域计算，传统方法与深度学习方法口径一致）

    参数:
        H_true: 真实信道矩阵（复数）
        H_est: 估计的信道矩阵（复数）

    返回:
        包含 'mse'、'nmse'、'nmse_db' 的字典；
        BER由 utils.ber.simulate_qpsk_ber 单独仿真（真实调制解调），不在此计算
    """
    mse = float(np.mean(np.abs(H_true - H_est) ** 2))
    signal_power = float(np.mean(np.abs(H_true) ** 2))
    nmse = mse / signal_power if signal_power > 0 else float('inf')

    return {
        'mse': mse,
        'nmse': float(nmse),
        'nmse_db': float(10 * np.log10(nmse)) if nmse > 0 else -float('inf')
    }
