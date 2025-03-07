import numpy as np
from scipy import linalg

class LSChannelEstimator:
    """最小二乘(LS)信道估计器"""
    
    def __init__(self):
        self.H_est = None  # 估计的信道矩阵
        
    def estimate(self, Y, X):
        """
        使用最小二乘方法估计信道
        
        参数:
            Y: 接收信号矩阵
            X: 发送信号矩阵（导频）
            
        返回:
            H_est: 估计的信道矩阵
        """
        # 使用最小二乘方法估计信道
        # H = Y * X^H * (X * X^H)^(-1)
        X_H = np.conjugate(X).T
        self.H_est = Y @ X_H @ linalg.inv(X @ X_H)
        return self.H_est
    
    def get_mse(self, H_true):
        """
        计算均方误差
        
        参数:
            H_true: 真实信道矩阵
            
        返回:
            mse: 均方误差
        """
        if self.H_est is None:
            raise ValueError("请先进行信道估计")
        
        mse = np.mean(np.abs(H_true - self.H_est) ** 2)
        return mse 