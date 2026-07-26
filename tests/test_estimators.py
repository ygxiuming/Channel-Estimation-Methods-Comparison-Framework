"""
信道估计框架合理性测试

验证修复后的核心正确性：
1. 无噪声时 LS/ML 能精确恢复信道
2. 高 SNR 时 LMMSE 收敛于 LS
3. 有噪声时 LMMSE 不劣于 LS
4. BER 随 SNR 单调（近似）下降，且完美信道下高SNR时BER≈0
5. 数据生成满足 Y = H·X + N 及噪声方差定义
6. 预处理缩放保持线性关系；同步增强保持 Y'=H'X+N' 统计一致
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from traditional.estimators import (LSEstimator, LMMSEEstimator, MLEstimator,
                                    calculate_performance_metrics)
from utils.data_generator import ChannelDataGenerator
from utils.preprocessing import ChannelPreprocessor, augment_triplets
from utils.ber import simulate_qpsk_ber, qpsk_modulate, qpsk_demodulate


@pytest.fixture
def generator():
    return ChannelDataGenerator(n_tx=4, n_rx=4, n_pilot=16,
                                channel_type='rayleigh')


@pytest.fixture
def clean_data(generator):
    """无噪声数据（snr=100dB 近似无噪）"""
    np.random.seed(0)
    return generator.generate(64, snr_db=100.0)


class TestTraditionalEstimators:
    def test_ls_exact_recovery_noiseless(self, generator):
        """无噪声时 LS 应精确恢复信道"""
        np.random.seed(1)
        H = generator.generate_channel(32)
        X = generator.generate_pilot_symbols(32)
        Y = H @ X  # 无噪声
        H_est = LSEstimator(regularization=False).estimate(Y, X)
        assert np.max(np.abs(H - H_est)) < 1e-10

    def test_ml_equals_ls(self, generator):
        """ML 闭式解应与无正则 LS 完全一致"""
        np.random.seed(2)
        H, X, Y, _ = generator.generate(32, snr_db=10.0)
        H_ml = MLEstimator().estimate(Y, X)
        H_ls = LSEstimator(regularization=False).estimate(Y, X)
        assert np.allclose(H_ml, H_ls, atol=1e-10)

    def test_lmmse_not_worse_than_ls(self, generator):
        """有噪声时 LMMSE 的 MSE 不应劣于 LS（统计意义）"""
        np.random.seed(3)
        H, X, Y, noise_var = generator.generate(2000, snr_db=10.0)
        mse_ls = calculate_performance_metrics(
            H, LSEstimator().estimate(Y, X))['mse']
        mse_lmmse = calculate_performance_metrics(
            H, LMMSEEstimator(noise_var=noise_var).estimate(Y, X))['mse']
        assert mse_lmmse <= mse_ls * 1.05  # 容忍5%数值波动

    def test_lmmse_approaches_ls_at_high_snr(self, generator):
        """高 SNR 时 LMMSE 应收敛于 LS"""
        np.random.seed(4)
        H, X, Y, noise_var = generator.generate(256, snr_db=40.0)
        H_ls = LSEstimator(regularization=False).estimate(Y, X)
        H_lmmse = LMMSEEstimator(noise_var=noise_var).estimate(Y, X)
        rel = np.mean(np.abs(H_lmmse - H_ls) ** 2) / np.mean(np.abs(H_ls) ** 2)
        assert rel < 1e-2

    def test_lmmse_matches_naive_per_sample(self, generator):
        """向量化 LMMSE 应与逐样本显式公式一致"""
        np.random.seed(5)
        H, X, Y, noise_var = generator.generate(16, snr_db=10.0)
        H_vec = LMMSEEstimator(noise_var=noise_var).estimate(Y, X)

        H_ls = LSEstimator(regularization=False).estimate(Y, X)
        n_tx = X.shape[1]
        H_naive = np.zeros_like(H_ls)
        for i in range(Y.shape[0]):
            XX_H = X[i] @ X[i].conj().T
            A = np.eye(n_tx) + noise_var * np.linalg.inv(XX_H)
            W = np.eye(n_tx) @ np.linalg.inv(A)
            for r in range(H.shape[1]):
                H_naive[i, r] = W @ H_ls[i, r]
        assert np.allclose(H_vec, H_naive, atol=1e-8)

    def test_metrics_complex_domain(self):
        """指标在复数域计算：MSE = mean|Δh|²"""
        H_true = np.array([[[1 + 1j, 2 - 1j]]])
        H_est = np.array([[[1 - 1j, 2 + 1j]]])
        m = calculate_performance_metrics(H_true, H_est)
        # |Δh|² = |2j|² = 4 和 |-2j|² = 4，MSE=4
        assert m['mse'] == pytest.approx(4.0)
        # 信号功率 = mean(|1+1j|², |2-1j|²) = mean(2, 5) = 3.5
        assert m['nmse'] == pytest.approx(4.0 / 3.5)


class TestDataGenerator:
    def test_shapes_and_noise_var(self, generator):
        np.random.seed(6)
        H, X, Y, noise_var = generator.generate(100, snr_db=10.0)
        assert H.shape == (100, 4, 4)
        assert X.shape == (100, 4, 16)
        assert Y.shape == (100, 4, 16)
        assert noise_var > 0

    def test_noise_variance_matches_snr(self, generator):
        """实测噪声功率应与 signal_power/snr 一致"""
        np.random.seed(7)
        H = generator.generate_channel(2000)
        X = generator.generate_pilot_symbols(2000)
        snr_db = 10.0
        Y, noise_var = generator.generate_received_signal(H, X, snr_db)
        N = Y - H @ X
        measured = np.mean(np.abs(N) ** 2)
        assert measured == pytest.approx(noise_var, rel=0.1)
        signal_power = np.mean(np.abs(H @ X) ** 2)
        assert noise_var == pytest.approx(signal_power / 10 ** (snr_db / 10), rel=0.1)

    def test_rician_unit_power(self):
        """Rician 信道应保持单位平均功率 E|h|²=1"""
        np.random.seed(8)
        gen = ChannelDataGenerator(4, 4, 16, channel_type='rician', rician_k=4.0)
        H = gen.generate_channel(20000)
        assert np.mean(np.abs(H) ** 2) == pytest.approx(1.0, rel=0.02)


class TestPreprocessing:
    def test_scaling_preserves_linear_relation(self, generator):
        """同一缩放因子作用于 H 和 Y 时，Y=HX+N 精确保持"""
        np.random.seed(9)
        H, X, Y, _ = generator.generate(32, snr_db=20.0)
        pre = ChannelPreprocessor('power')
        pre.fit(H)
        H_s, Y_s = pre.transform(H), pre.transform(Y)
        N_orig = Y - H @ X
        N_scaled = Y_s - H_s @ X
        assert np.allclose(N_scaled, N_orig / pre.scale, atol=1e-12)

    def test_augmentation_consistency(self, generator):
        """同步增强后 Y' = H'·X + N' 且 N' 与 N 统计一致（同分布同变换）"""
        np.random.seed(10)
        H, X, Y, _ = generator.generate(64, snr_db=20.0)
        N = Y - H @ X
        np.random.seed(11)
        H2, X2, Y2 = augment_triplets(H, X, Y, methods=['phase_shift'])
        N2 = Y2 - H2 @ X
        # 相位旋转下 |N'| = |N|
        assert np.allclose(np.abs(N2), np.abs(N), atol=1e-10)
        assert np.allclose(X2, X)


class TestBER:
    def test_modulate_demodulate_roundtrip(self):
        bits = np.array([[0, 0, 0, 1, 1, 0, 1, 1]])
        assert np.array_equal(qpsk_demodulate(qpsk_modulate(bits)), bits)

    def test_ber_decreases_with_snr(self):
        """BER 应随 SNR 单调（统计意义）下降"""
        np.random.seed(12)
        gen = ChannelDataGenerator(4, 4, 16)
        H = gen.generate_channel(200)
        H_est = H  # 完美信道估计
        bers = [simulate_qpsk_ber(H, H_est, snr, n_symbols=200,
                                  rng=np.random.default_rng(42))
                for snr in [0, 10, 20]]
        assert bers[0] > bers[1] > bers[2]

    def test_ber_zero_at_high_snr_perfect_channel(self):
        """完美估计 + 高SNR 时 BER 应接近 0"""
        np.random.seed(13)
        gen = ChannelDataGenerator(4, 4, 16)
        H = gen.generate_channel(100)
        ber = simulate_qpsk_ber(H, H, 30, n_symbols=200,
                                rng=np.random.default_rng(7))
        assert ber < 1e-3

    def test_ber_worse_with_noisy_estimate(self):
        """有噪估计的 BER 应差于完美估计"""
        np.random.seed(14)
        gen = ChannelDataGenerator(4, 4, 16)
        H = gen.generate_channel(200)
        noise = (np.random.normal(0, 0.2, H.shape) +
                 1j * np.random.normal(0, 0.2, H.shape))
        ber_perfect = simulate_qpsk_ber(H, H, 10, n_symbols=200,
                                        rng=np.random.default_rng(1))
        ber_noisy = simulate_qpsk_ber(H, H + noise, 10, n_symbols=200,
                                      rng=np.random.default_rng(1))
        assert ber_noisy > ber_perfect


class TestDLModels:
    """深度学习模型的维度正确性"""

    @pytest.mark.parametrize('n_tx,n_rx,n_pilot', [(4, 4, 16), (8, 4, 16), (2, 3, 8)])
    @pytest.mark.parametrize('mode', ['raw', 'ls_refine'])
    def test_output_shapes_any_config(self, n_tx, n_rx, n_pilot, mode):
        """任意天线/导频配置下输出维度都应为 (B, n_rx, 2*n_tx)"""
        import torch
        from deep_learning.models import (CNNEstimator, RNNEstimator, LSTMEstimator,
                                          GRUEstimator, HybridEstimator)
        if mode == 'raw':
            input_size = 2 * n_pilot * (n_rx + n_tx)
        else:
            input_size = 2 * n_rx * n_tx

        for cls, kwargs in [
            (CNNEstimator, {}),
            (RNNEstimator, {'hidden_size': 32, 'num_layers': 1}),
            (LSTMEstimator, {'hidden_size': 32, 'num_layers': 1}),
            (GRUEstimator, {'hidden_size': 32, 'num_layers': 1}),
            (HybridEstimator, {'hidden_size': 32}),
        ]:
            model = cls(input_size=input_size, n_rx=n_rx, n_tx=n_tx, **kwargs)
            x = torch.randn(5, input_size)
            out = model(x)
            assert out.shape == (5, n_rx, 2 * n_tx), \
                f"{cls.__name__}({mode}) 输出形状错误: {out.shape}"
