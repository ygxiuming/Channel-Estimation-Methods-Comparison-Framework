"""
真实误比特率(BER)仿真模块

对每种信道估计结果，通过完整的通信链路仿真统计误比特率：
    随机比特 -> QPSK调制 -> 真实信道H -> 加性高斯白噪声
    -> 用估计信道H_est做均衡(ZF/MMSE) -> QPSK判决 -> 统计误比特

所有估计方法（传统+深度学习）共用本模块，保证BER口径一致、对比公平。
"""

import numpy as np
from typing import Optional


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    QPSK调制（Gray映射，单位符号功率）

    参数:
        bits: 0/1比特数组，长度为偶数，shape=(..., 2*n_sym)
    返回:
        QPSK符号，shape=(..., n_sym)
    """
    bits = np.asarray(bits)
    b_i = bits[..., 0::2]
    b_q = bits[..., 1::2]
    return ((1 - 2 * b_i) + 1j * (1 - 2 * b_q)) / np.sqrt(2)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """
    QPSK硬判决解调（qpsk_modulate的逆操作）

    参数:
        symbols: 均衡后的符号，shape=(..., n_sym)
    返回:
        0/1比特数组，shape=(..., 2*n_sym)
    """
    b_i = (np.real(symbols) < 0).astype(int)
    b_q = (np.imag(symbols) < 0).astype(int)
    bits = np.stack([b_i, b_q], axis=-1)
    return bits.reshape(bits.shape[:-2] + (-1,))


def simulate_qpsk_ber(H_true: np.ndarray, H_est: np.ndarray, snr_db: float,
                      n_symbols: int = 100, equalizer: str = 'mmse',
                      rng: Optional[np.random.Generator] = None) -> float:
    """
    通过真实QPSK链路仿真计算误比特率（批量向量化）

    参数:
        H_true: 真实信道矩阵，shape=(n_samples, n_rx, n_tx)
        H_est: 估计的信道矩阵，shape=(n_samples, n_rx, n_tx)
        snr_db: 信噪比（dB），定义为接收端每元素平均信号功率/噪声功率
        n_symbols: 每个样本、每根发射天线发送的QPSK符号数
        equalizer: 均衡方式，'zf'（迫零，要求n_rx>=n_tx）或 'mmse'
        rng: 可选的随机数生成器（用于结果可复现）

    返回:
        误比特率（0~0.5之间的浮点数）
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples, n_rx, n_tx = H_true.shape
    if equalizer == 'zf' and n_rx < n_tx:
        raise ValueError("ZF均衡要求 n_rx >= n_tx")

    # 1) 生成随机比特并调制为QPSK符号（单天线单位符号功率）
    bits = rng.integers(0, 2, size=(n_samples, n_tx, 2 * n_symbols))
    S = qpsk_modulate(bits)                              # (n, n_tx, n_sym)

    # 2) 通过真实信道: Y_data = H_true @ S
    Y_data = np.matmul(H_true, S)                        # (n, n_rx, n_sym)

    # 3) 加噪（与数据生成器相同的SNR定义）
    snr = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(Y_data) ** 2)
    noise_var = signal_power / snr
    noise = (rng.normal(0, np.sqrt(noise_var / 2), Y_data.shape) +
             1j * rng.normal(0, np.sqrt(noise_var / 2), Y_data.shape))
    Y_data = Y_data + noise

    # 4) 用估计的信道做均衡
    H_est_H = np.conjugate(np.transpose(H_est, (0, 2, 1)))  # (n, n_tx, n_rx)
    if equalizer == 'zf':
        # 伪逆: S_est = pinv(H_est) @ Y
        G = np.linalg.pinv(H_est)                        # (n, n_tx, n_rx)
    elif equalizer == 'mmse':
        # MMSE均衡: G = (H_estᴴ·H_est + noise_var·I)⁻¹·H_estᴴ
        gram = np.matmul(H_est_H, H_est) + noise_var * np.eye(n_tx)[np.newaxis]
        G = np.linalg.solve(gram, H_est_H)
    else:
        raise ValueError("equalizer必须是'zf'或'mmse'")

    S_est = np.matmul(G, Y_data)                         # (n, n_tx, n_sym)

    # 5) 判决并统计误比特
    bits_est = qpsk_demodulate(S_est)
    ber = float(np.mean(bits != bits_est))

    return ber


def simulate_qpsk_ber_ofdm(H_true: np.ndarray, H_est: np.ndarray, snr_db: float,
                           n_symbols: int = 40, equalizer: str = 'mmse',
                           rng: Optional[np.random.Generator] = None) -> float:
    """
    OFDM 频选信道的 QPSK 链路 BER 仿真（逐子载波 MIMO 均衡）

    参数:
        H_true: 真实频域信道 (n_samples, n_rx, n_sc, n_tx)
        H_est:  估计频域信道，形状同上
        snr_db: 与数据生成器一致的 SNR 口径（全符号平均信号功率/噪声功率）
        n_symbols: 每个样本、每 (tx, 子载波) 发送的 QPSK 符号数
        equalizer: 'zf' 或 'mmse'（逐子载波均衡）
        rng: 随机数生成器（复现实验）

    返回: 误比特率
    """
    if rng is None:
        rng = np.random.default_rng()
    n, n_rx, n_sc, n_tx = H_true.shape
    if equalizer == 'zf' and n_rx < n_tx:
        raise ValueError("ZF均衡要求 n_rx >= n_tx")

    # 1) 全频域数据符号：bits (n, n_tx, n_sc, 2·n_sym) → S (n, n_tx, n_sc, n_sym)
    bits = rng.integers(0, 2, size=(n, n_tx, n_sc, 2 * n_symbols))
    S = qpsk_modulate(bits)

    # 2) 通过真实信道 + 全局噪声口径（与生成器一致：全符号平均功率定方差）
    Y_clean = np.einsum('nrkt,ntkm->nrkm', H_true, S)          # (n, n_rx, n_sc, n_sym)
    noise_var = float(np.mean(np.abs(Y_clean) ** 2) / 10 ** (snr_db / 10))
    noise = (rng.normal(0, np.sqrt(noise_var / 2), Y_clean.shape) +
             1j * rng.normal(0, np.sqrt(noise_var / 2), Y_clean.shape))
    Y = Y_clean + noise

    # 3) 逐子载波均衡：该 (k) 上是 n_rx×n_tx 的平坦 MIMO
    bits_est = np.zeros_like(bits)
    eye = np.eye(n_tx)
    for k in range(n_sc):
        Hk_est = H_est[:, :, k, :]                                  # (n, n_rx, n_tx)
        Hk_H = np.conjugate(np.transpose(Hk_est, (0, 2, 1)))
        if equalizer == 'zf':
            G = np.linalg.pinv(Hk_est)                              # (n, n_tx, n_rx)
        else:
            gram = np.matmul(Hk_H, Hk_est) + noise_var * eye[np.newaxis]
            G = np.linalg.solve(gram, Hk_H)                         # (n, n_tx, n_rx)
        S_est = np.matmul(G, Y[:, :, k, :])                         # (n, n_tx, n_sym)
        bits_est[:, :, k, :] = qpsk_demodulate(S_est)

    return float(np.mean(bits != bits_est))
