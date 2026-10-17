"""
深度学习输入特征构造模块

统一约定（与 deep_learning.models 的说明保持一致）：
- 输入: 展平的实数特征向量 (batch, input_size)，由 build_dl_inputs 按输入模式构造
- 输出: (batch, n_rx, 2·n_tx)，最后一维为 [实部 | 虚部] 拼接

支持的输入模式 (input modes):
- 'raw'         : 块拼接（历史方式）——[Re(Y)|Im(Y)|Re(X)|Im(X)] 四大块顺序展平。
                  缺陷：CNN 的 3×1 卷积核与 RNN 的伪时序切分都局限在"同一块内部"，
                  网络局部感受野永远看不到 (y, x) 配对，难以学习 H ≈ Y·X⁻¹ 的乘性结构。
- 'raw_sym'     : 按导频符号重排（改进方式）——组织为 (n, n_pilot, 2·n_rx + 2·n_tx)
                  后展平，每个时间步 = 同一导频符号上的 [y(各接收天线), x(各发射天线)]。
                  时序/卷积的每一步都有完整的配对语义，与 RNN/LSTM/GRU/卷积的结构匹配。
- 'ls_refine'   : 混合式（历史方式）——仅输入 [Re(H_ls), Im(H_ls)]。
                  缺陷：LS 误差的协方差为 noise_var·(X·Xᴴ)⁻¹，逐样本不同（取决于 X），
                  网络看不到 X 就学不出逐样本最优的收缩/去噪，只能学全局平均收缩，
                  高 SNR 时对本已很好的 LS 估计过度收缩（引入偏差）。
- 'ls_refine_x' : 混合式 + 导频特征（改进方式）——在 ls_refine 基础上追加
                  [Re(X), Im(X)]，使网络可见逐样本噪声协方差的全部决定因素。

所有维度（n_rx, n_tx, n_pilot）由数组形状显式传入，不做任何隐式假设。
"""

import numpy as np

from traditional.estimators import LSEstimator

# 支持的输入模式
INPUT_MODES = ['raw', 'raw_sym', 'ls_refine', 'ls_refine_x']

# 各模式的中文说明（用于控制台面板与实验报告）
MODE_DESCRIPTIONS = {
    'raw': '端到端块拼接：[Re(Y)|Im(Y)|Re(X)|Im(X)]（历史方式）',
    'raw_sym': '端到端按导频符号重排：每时间步 = (y, x) 配对（改进）',
    'ls_refine': '混合式：仅输入 LS 估计实虚部（历史方式）',
    'ls_refine_x': '混合式：LS 估计 + 导频特征，网络可见噪声协方差信息（改进）',
}


def to_complex(pred: np.ndarray, n_tx: int) -> np.ndarray:
    """将网络输出 (n, n_rx, 2*n_tx) 的 [实部|虚部] 拼接重组为复数信道"""
    return pred[..., :n_tx] + 1j * pred[..., n_tx:]


def complex_to_real_tensor(H: np.ndarray) -> np.ndarray:
    """将复数信道 (n, n_rx, n_tx) 转为 (n, n_rx, 2*n_tx) 实数目标张量"""
    return np.concatenate([H.real, H.imag], axis=-1).astype(np.float32)


def get_input_size(mode: str, n_rx: int, n_tx: int, n_pilot: int) -> int:
    """返回指定输入模式的特征维度（与 build_dl_inputs 的输出维一致）"""
    if mode in ('raw', 'raw_sym'):
        return 2 * n_pilot * (n_rx + n_tx)
    if mode in ('ls_refine', 'ls_refine_x'):
        return 2 * n_rx * n_tx + (2 * n_tx * n_pilot if mode == 'ls_refine_x' else 0)
    raise ValueError(f"未知的DL输入模式: {mode}，可选 {INPUT_MODES}")


def build_dl_inputs(H: np.ndarray, X: np.ndarray, Y: np.ndarray,
                    mode: str) -> np.ndarray:
    """
    构造深度学习模型的输入特征（输入必须携带信道信息）

    参数:
        H: 真实信道 (n, n_rx, n_tx)（仅 'ls_refine*' 模式间接受用于计算 LS）
        X: 导频符号 (n, n_tx, n_pilot)
        Y: 接收信号 (n, n_rx, n_pilot)——唯一直接携带 H 信息的观测量
        mode: 'raw' / 'raw_sym' / 'ls_refine' / 'ls_refine_x'

    返回:
        展平的实数输入特征 (n, input_size)
    """
    if mode not in INPUT_MODES:
        raise ValueError(f"未知的DL输入模式: {mode}，可选 {INPUT_MODES}")

    n = H.shape[0]

    if mode in ('raw', 'raw_sym'):
        if mode == 'raw':
            # 历史方式：四大块顺序拼接
            feats = np.concatenate([
                Y.real.reshape(n, -1), Y.imag.reshape(n, -1),
                X.real.reshape(n, -1), X.imag.reshape(n, -1)
            ], axis=1)
        else:
            # 改进方式：按导频符号组织
            # Y: (n, n_rx, n_pilot) -> (n, n_pilot, n_rx)
            # X: (n, n_tx, n_pilot) -> (n, n_pilot, n_tx)
            y_re = np.transpose(Y.real, (0, 2, 1))
            y_im = np.transpose(Y.imag, (0, 2, 1))
            x_re = np.transpose(X.real, (0, 2, 1))
            x_im = np.transpose(X.imag, (0, 2, 1))
            # (n, n_pilot, 2*n_rx + 2*n_tx) -> (n, input_size)
            feats = np.concatenate([y_re, y_im, x_re, x_im], axis=2).reshape(n, -1)

    elif mode in ('ls_refine', 'ls_refine_x'):
        H_ls = LSEstimator(regularization=False).estimate(Y, X)
        feats = np.concatenate([
            H_ls.real.reshape(n, -1), H_ls.imag.reshape(n, -1)
        ], axis=1)
        if mode == 'ls_refine_x':
            # 追加导频特征：网络由此可学习逐样本噪声协方差 noise_var·(XXᴴ)⁻¹
            feats = np.concatenate([
                feats,
                X.real.reshape(n, -1), X.imag.reshape(n, -1)
            ], axis=1)

    return feats.astype(np.float32)
