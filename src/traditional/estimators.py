"""
传统信道估计方法模块
包含LS、LMMSE和ML估计器
"""

import numpy as np
from scipy.special import erf
from typing import Dict, Union, Optional

class LSEstimator:
    """最小二乘(LS)估计器"""
    
    def __init__(self, regularization: bool = False, lambda_: float = 0.01):
        """
        初始化LS估计器
        
        参数:
            regularization: 是否使用正则化
            lambda_: 正则化参数
        """
        self.regularization = regularization
        self.lambda_ = lambda_
    
    def estimate(self, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        执行LS估计
        
        参数:
            Y: 接收信号，shape=(n_samples, n_rx, n_pilot)
            X: 导频符号，shape=(n_samples, n_tx, n_pilot)
        
        返回:
            H_est: 估计的信道矩阵，shape=(n_samples, n_rx, n_tx)
        """
        # 转置X以便进行批量矩阵运算
        X_H = np.conjugate(np.transpose(X, (0, 2, 1)))  # (n_samples, n_pilot, n_tx)
        
        # 计算XX^H
        XX_H = np.matmul(X, X_H)  # (n_samples, n_tx, n_tx)
        
        if self.regularization:
            # 添加正则化项
            n_tx = X.shape[1]
            reg_term = self.lambda_ * np.eye(n_tx)[np.newaxis, :, :]
            XX_H = XX_H + reg_term
        
        # 使用SVD进行稳定的矩阵求逆
        H_est_list = []
        for i in range(Y.shape[0]):
            try:
                # 尝试直接求逆
                XX_H_inv = np.linalg.inv(XX_H[i])
            except np.linalg.LinAlgError:
                # 如果矩阵接近奇异，使用伪逆
                XX_H_inv = np.linalg.pinv(XX_H[i])
            
            # 计算信道估计
            H_est_i = np.matmul(Y[i], np.matmul(X_H[i], XX_H_inv))
            H_est_list.append(H_est_i)
        
        H_est = np.stack(H_est_list, axis=0)
        return H_est

class LMMSEEstimator:
    """线性最小均方误差(LMMSE)估计器"""
    
    def __init__(self, snr: float, adaptive_snr: bool = False,
                 correlation_method: str = 'sample'):
        """
        初始化LMMSE估计器
        
        参数:
            snr: 信噪比（线性，非dB）
            adaptive_snr: 是否使用自适应SNR估计
            correlation_method: 信道相关矩阵估计方法，'sample' 或 'theoretical'
        """
        self.snr = snr
        self.adaptive_snr = adaptive_snr
        self.correlation_method = correlation_method
        
        if correlation_method not in ['sample', 'theoretical']:
            raise ValueError("correlation_method必须是'sample'或'theoretical'")
    
    def _estimate_correlation(self, H_ls: np.ndarray) -> np.ndarray:
        """
        估计信道相关矩阵
        
        参数:
            H_ls: LS估计的信道矩阵
        
        返回:
            R_H: 信道相关矩阵
        """
        if self.correlation_method == 'sample':
            # 使用样本相关矩阵
            H_ls_flat = H_ls.reshape(H_ls.shape[0], -1)
            R_H = np.matmul(H_ls_flat.T.conj(), H_ls_flat) / H_ls.shape[0]
            # 添加小的正则化项以提高数值稳定性
            epsilon = 1e-10
            R_H = R_H + epsilon * np.eye(R_H.shape[0])
        else:
            # 使用理论相关矩阵（假设指数衰减）
            n_rx, n_tx = H_ls.shape[1:3]
            R_H = np.zeros((n_rx * n_tx, n_rx * n_tx), dtype=complex)
            rho = 0.7  # 相关系数
            for i in range(n_rx * n_tx):
                for j in range(n_rx * n_tx):
                    R_H[i, j] = rho ** abs(i - j)
        
        return R_H
    
    def estimate(self, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        执行LMMSE估计
        
        参数:
            Y: 接收信号，shape=(n_samples, n_rx, n_pilot)
            X: 导频符号，shape=(n_samples, n_tx, n_pilot)
        
        返回:
            H_est: 估计的信道矩阵，shape=(n_samples, n_rx, n_tx)
        """
        # 首先进行LS估计
        ls_estimator = LSEstimator(regularization=True, lambda_=0.01)  # 使用正则化的LS估计
        H_ls = ls_estimator.estimate(Y, X)
        
        # 估计信道相关矩阵
        R_H = self._estimate_correlation(H_ls)
        
        if self.adaptive_snr:
            # 使用样本方差估计噪声功率
            error = Y - np.matmul(H_ls, X)
            noise_power = np.mean(np.abs(error) ** 2)
            signal_power = np.mean(np.abs(Y) ** 2)
            snr = max(signal_power / noise_power, 1.0)  # 确保SNR不小于1
        else:
            snr = self.snr
        
        # LMMSE估计
        n_samples = Y.shape[0]
        H_est = np.zeros_like(H_ls)
        
        for i in range(n_samples):
            h_ls = H_ls[i].flatten()
            # 使用更稳定的求解方法
            try:
                # 使用Cholesky分解求解线性方程组
                A = R_H + np.eye(len(h_ls)) / snr
                L = np.linalg.cholesky(A)
                h_est = h_ls.copy()
                # 解线性方程组 A @ x = R_H @ h_ls
                y = np.linalg.solve(L, R_H @ h_ls)
                h_est = np.linalg.solve(L.T.conj(), y)
            except np.linalg.LinAlgError:
                # 如果Cholesky分解失败，使用伪逆
                h_est = np.matmul(R_H, np.linalg.pinv(R_H + np.eye(len(h_ls)) / snr)) @ h_ls
            
            H_est[i] = h_est.reshape(H_ls.shape[1:])
        
        return H_est

class MLEstimator:
    """最大似然(ML)估计器"""
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-6, learning_rate: float = 0.01):
        """
        初始化ML估计器
        
        参数:
            max_iter: 最大迭代次数
            tol: 收敛阈值
            learning_rate: 初始学习率
        """
        self.max_iter = max_iter
        self.tol = float(tol)
        self.learning_rate = learning_rate
        self.min_lr = 1e-6  # 最小学习率
        self.lr_decay = 0.95  # 学习率衰减因子
    
    def _calculate_loss(self, H_est: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
        """计算损失函数（负对数似然）"""
        error = Y - np.matmul(H_est, X)
        return np.mean(np.abs(error) ** 2)
    
    def estimate(self, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        执行ML估计
        
        参数:
            Y: 接收信号，shape=(n_samples, n_rx, n_pilot)
            X: 导频符号，shape=(n_samples, n_tx, n_pilot)
        
        返回:
            H_est: 估计的信道矩阵，shape=(n_samples, n_rx, n_tx)
        """
        # 使用正则化的LS估计作为初始值
        ls_estimator = LSEstimator(regularization=True, lambda_=0.01)
        H_est = ls_estimator.estimate(Y, X)
        
        # 迭代优化
        X_H = np.conjugate(np.transpose(X, (0, 2, 1)))  # (n_samples, n_pilot, n_tx)
        lr = self.learning_rate
        best_H = H_est.copy()
        best_loss = self._calculate_loss(H_est, X, Y)
        
        for iter_idx in range(self.max_iter):
            H_prev = H_est.copy()
            prev_loss = self._calculate_loss(H_prev, X, Y)
            
            # 计算误差
            error = Y - np.matmul(H_est, X)  # (n_samples, n_rx, n_pilot)
            
            # 计算梯度
            gradient = -np.matmul(error, X_H)  # 负梯度方向
            
            # 梯度缩放
            grad_norm = np.maximum(np.sqrt(np.mean(np.abs(gradient) ** 2)), 1e-8)
            gradient = gradient / grad_norm
            
            # 更新估计
            H_est = H_est - lr * gradient
            
            # 计算当前损失
            current_loss = self._calculate_loss(H_est, X, Y)
            
            # 如果损失增加，回退并降低学习率
            if current_loss > prev_loss:
                H_est = H_prev
                lr = max(lr * self.lr_decay, self.min_lr)
                continue
            
            # 更新最佳结果
            if current_loss < best_loss:
                best_H = H_est.copy()
                best_loss = current_loss
            
            # 检查收敛
            diff = float(np.mean(np.abs(H_est - H_prev) ** 2))
            if diff < self.tol:
                break
            
            # 定期降低学习率
            if iter_idx > 0 and iter_idx % 10 == 0:
                lr = max(lr * self.lr_decay, self.min_lr)
        
        return best_H

def calculate_performance_metrics(H_true: np.ndarray, H_est: np.ndarray) -> Dict[str, float]:
    """
    计算性能指标
    
    参数:
        H_true: 真实信道矩阵
        H_est: 估计的信道矩阵
    
    返回:
        包含各种性能指标的字典
    """
    # 计算MSE
    mse = np.mean(np.abs(H_true - H_est) ** 2)
    
    # 计算NMSE
    nmse = mse / np.mean(np.abs(H_true) ** 2)
    
    # 计算BER（假设QPSK调制）
    def qpsk_ber(h_true, h_est):
        # 添加小的常数以避免除零
        epsilon = 1e-10
        snr_eff = np.abs(h_true) ** 2 / (np.abs(h_true - h_est) ** 2 + epsilon)
        return 0.5 * np.mean(1 - erf(np.sqrt(snr_eff/2)))
    
    ber = qpsk_ber(H_true, H_est)
    
    return {
        'mse': float(mse),
        'nmse': float(nmse),
        'ber': float(ber)
    } 