"""
输入特征构造与盲噪声方差估计测试

1. 四种输入模式均可构造，维度正确且与 get_input_size 一致
2. 'raw_sym' 按导频符号重排：与手动构造逐元素一致
   （每个时间步 = 同一导频符号上的 [y(接收天线), x(发射天线)] 配对）
3. 'ls_refine_x' 在 'ls_refine' 的特征基础上追加导频实虚部
4. 盲噪声方差估计无偏、精度足够（不使用上帝信息）
5. 盲噪声方差驱动的 LMMSE 在低 SNR 仍不劣于 LS（统计意义）
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from traditional.estimators import (LSEstimator, LMMSEEstimator,
                                    estimate_noise_var_blind)
from utils.data_generator import ChannelDataGenerator
from utils.features import (build_dl_inputs, get_input_size, to_complex,
                            complex_to_real_tensor, INPUT_MODES)


@pytest.fixture
def generator():
    return ChannelDataGenerator(n_tx=4, n_rx=4, n_pilot=16,
                                channel_type='rayleigh')


@pytest.fixture
def data(generator):
    np.random.seed(0)
    return generator.generate(32, snr_db=10.0)


class TestFeatureBuilding:
    def test_all_modes_buildable_correct_shape(self, generator, data):
        """四种模式都能构造，特征维度与 get_input_size 一致"""
        H, X, Y, _ = data
        expected = {
            'raw': 2 * 16 * (4 + 4),           # 256
            'raw_sym': 2 * 16 * (4 + 4),       # 与 raw 同维度
            'ls_refine': 2 * 4 * 4,            # 32
            'ls_refine_x': 2 * 4 * 4 + 2 * 4 * 16,   # 160
        }
        for mode in INPUT_MODES:
            feats = build_dl_inputs(H, X, Y, mode)
            assert feats.shape == (32, expected[mode])
            assert feats.shape[1] == get_input_size(mode, 4, 4, 16)
            assert feats.dtype == np.float32
            assert np.all(np.isfinite(feats))

    def test_raw_sym_layout_matches_pilotsymbol_pairing(self, generator, data):
        """raw_sym 每个时间步 = 同一导频符号上的 (y,x) 配对，与手动构造一致"""
        H, X, Y, _ = data
        feats = build_dl_inputs(H, X, Y, 'raw_sym').reshape(32, 16, -1)

        y_re = np.transpose(Y.real, (0, 2, 1))   # (n, n_pilot, n_rx)
        y_im = np.transpose(Y.imag, (0, 2, 1))
        x_re = np.transpose(X.real, (0, 2, 1))   # (n, n_pilot, n_tx)
        x_im = np.transpose(X.imag, (0, 2, 1))
        manual = np.concatenate([y_re, y_im, x_re, x_im], axis=2)

        assert np.allclose(feats, manual.astype(np.float32), atol=1e-7)

        # 检查配对语义：第 t 步的第 i 维 = 第 t 个导频符号上第 i 根接收天线的 Re(y)
        t, i = 3, 2
        assert np.allclose(feats[:, t, i], Y[:, i, t].real.astype(np.float32))
        # 第 t 步的后半部分 = 同一符号上的发射天线导频
        assert np.allclose(feats[:, t, 2 * 4 + 1],
                           X[:, 1, t].real.astype(np.float32))

    def test_raw_sym_same_size_as_raw(self, generator, data):
        """raw_sym 与 raw 输入维度相同，仅语义重排（网络结构无需变化即可对比）"""
        H, X, Y, _ = data
        assert (build_dl_inputs(H, X, Y, 'raw').shape ==
                build_dl_inputs(H, X, Y, 'raw_sym').shape)

    def test_ls_refine_x_appends_pilot_features(self, generator, data):
        """ls_refine_x = ls_refine 特征 + 导频实虚部"""
        H, X, Y, _ = data
        base = build_dl_inputs(H, X, Y, 'ls_refine')
        full = build_dl_inputs(H, X, Y, 'ls_refine_x')
        n_base = base.shape[1]
        assert np.allclose(full[:, :n_base], base, atol=1e-7)
        pilot_fts = np.concatenate(
            [X.real.reshape(32, -1), X.imag.reshape(32, -1)], axis=1)
        assert np.allclose(full[:, n_base:], pilot_fts.astype(np.float32),
                           atol=1e-7)

    def test_legacy_raw_layout_unchanged(self, generator, data):
        """'raw' 保持历史块拼接布局（向后兼容，旧结果可复现）"""
        H, X, Y, _ = data
        feats = build_dl_inputs(H, X, Y, 'raw')
        manual = np.concatenate([
            Y.reshape(32, -1).real, Y.reshape(32, -1).imag,
            X.reshape(32, -1).real, X.reshape(32, -1).imag], axis=1)
        assert np.allclose(feats, manual.astype(np.float32), atol=1e-7)

    def test_unknown_mode_raises(self, generator, data):
        H, X, Y, _ = data
        with pytest.raises(ValueError):
            build_dl_inputs(H, X, Y, 'transformer')

    def test_complex_roundtrip(self, generator, data):
        """to_complex 与 complex_to_real_tensor 互逆"""
        H, X, Y, _ = data
        tensor = complex_to_real_tensor(H)
        H_back = to_complex(tensor, 4)
        assert np.allclose(H_back, H, atol=1e-5)


class TestBlindNoiseVariance:
    def test_estimate_unbiased_and_accurate(self, generator):
        """盲估计值应接近生成器实际使用的噪声方差（LS 残差法为无偏估计）"""
        np.random.seed(1)
        for snr_db in (5.0, 25.0):
            generator2 = ChannelDataGenerator(n_tx=4, n_rx=4, n_pilot=16)
            # 大样本让残差统计量收敛到整体使用的那一个标量噪声功率
            n = 4000
            H = generator2.generate_channel(n)
            X = generator2.generate_pilot_symbols(n)
            np.random.seed(2)
            Y, true_var = generator2.generate_received_signal(H, X, snr_db)
            est = estimate_noise_var_blind(Y, X)
            rel_err = abs(est - true_var) / true_var
            assert rel_err < 0.05, (f"snr={snr_db}, est={est}, true={true_var}")

    def test_estimate_requires_pilot_gt_ntx(self):
        """n_pilot == n_tx 时 LS 残差自由度为零，应报错"""
        gen = ChannelDataGenerator(n_tx=4, n_rx=4, n_pilot=4)
        np.random.seed(3)
        H, X, Y, _ = gen.generate(16, snr_db=10.0)
        with pytest.raises(ValueError):
            estimate_noise_var_blind(Y, X)

    def test_lmmse_with_blind_var_beats_ls_at_low_snr(self, generator):
        """用盲估计噪声方差驱动的 LMMSE，在低 SNR 仍应优于 LS"""
        np.random.seed(4)
        H, X, Y, _ = generator.generate(1000, snr_db=5.0)
        var_blind = estimate_noise_var_blind(Y, X)
        lmmse = LMMSEEstimator(noise_var=var_blind,
                              correlation_method='theoretical')
        H_lmmse = lmmse.estimate(Y, X)
        H_ls = LSEstimator(regularization=False).estimate(Y, X)
        mse_lmmse = float(np.mean(np.abs(H - H_lmmse) ** 2))
        mse_ls = float(np.mean(np.abs(H - H_ls) ** 2))
        assert mse_lmmse < mse_ls, (mse_lmmse, mse_ls)
