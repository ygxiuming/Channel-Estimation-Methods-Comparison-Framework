"""
正交导频、OFDM 频选信道与注意力 unfolding 模型测试

1. 正交导频（dft/hadamard）：X·Xᴴ = (n_pilot/n_tx)·I，符号功率口径一致
2. OFDM 生成器：梳状导频结构、频域信道二阶统计与解析 PDP 相关一致、
   无噪时子载波观测 = 单天线路径 × 符号
3. OFDM 估计器：梳状点 LS 精确性（按各天线自己的栅格）、
   MMSE 频域插值优于线性插值、噪声免推时趋于精确
4. 特征层 4 维（频域网格）支持：build_dl_inputs / to_complex / get_input_size
5. LmmseNetAtt：前向/反传形状、第二段物理层梯度流通、输入校验
6. OFDM QPSK 链路 BER：genie 信道高 SNR 趋近 0，失效估计 BER 接近 0.5
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from traditional.estimators import (OFDMCombLSEstimator, OFDMLMMSEEstimator,
                                    calculate_performance_metrics)
from utils.data_generator import ChannelDataGenerator, OFDMDataGenerator
from utils.features import (build_dl_inputs, get_input_size, to_complex,
                            complex_to_real_tensor)
from utils.ber import simulate_qpsk_ber_ofdm


# ---------------------------------------------------------------- 正交导频

def test_dft_pilot_orthogonality():
    gen = ChannelDataGenerator(4, 4, 16, pilot_type='dft')
    X = gen.generate_pilot_symbols(8)
    XX_H = np.matmul(X, np.conj(np.transpose(X, (0, 2, 1))))
    ref = np.broadcast_to(4.0 * np.eye(4), XX_H.shape)
    np.testing.assert_allclose(XX_H, ref, atol=1e-10)


def test_hadamard_pilot_orthogonality():
    gen = ChannelDataGenerator(4, 4, 16, pilot_type='hadamard')
    X = gen.generate_pilot_symbols(8)
    XX_H = np.matmul(X, np.conj(np.transpose(X, (0, 2, 1))))
    ref = np.broadcast_to(4.0 * np.eye(4), XX_H.shape)
    np.testing.assert_allclose(XX_H, ref, atol=1e-10)
    # 每符号功率 1/n_tx（与随机导频口径一致）
    np.testing.assert_allclose(np.abs(X[0]) ** 2, 1 / 4, atol=1e-12)


def test_hadamard_requires_power_of_two():
    with pytest.raises(ValueError, match='hadamard'):
        ChannelDataGenerator(4, 4, 12, pilot_type='hadamard')


def test_pilot_signal_power_unchanged():
    """三种导频的接收信号功率口径一致：X 逐元素模方 = 1/n_tx"""
    for ptype in ('random', 'dft', 'hadamard'):
        gen = ChannelDataGenerator(4, 4, 16, pilot_type=ptype)
        X = gen.generate_pilot_symbols(32)
        assert np.allclose(np.abs(X) ** 2, 1 / 4, atol=0.15)


# ------------------------------------------------------------ OFDM 生成器

@pytest.fixture
def ofdm():
    return OFDMDataGenerator(n_tx=4, n_rx=4, n_sc=16, n_taps=4,
                             pdp_decay=0.35)


def test_ofdm_shapes_and_power(ofdm):
    H, X, Y, noise_var = ofdm.generate(64, 15.0)
    assert H.shape == (64, 4, 16, 4)
    assert X.shape == (64, 4, 16)
    assert Y.shape == (64, 4, 16)
    # 归一化指数 PDP ⇒ 频域信道元素平均功率 1（NMSE 口径与平坦信道一致）
    assert abs((np.abs(H) ** 2).mean() - 1.0) < 0.08


def test_ofdm_comb_structure(ofdm):
    """发射天线 t 的导频只在子载波 k ≡ t (mod n_tx) 非零"""
    X = ofdm.generate_pilot_symbols(4)
    for t in range(4):
        cols = np.arange(t, 16, 4)
        assert np.all(np.abs(X[:, t, cols]) > 0)
        mask = np.ones(16, bool)
        mask[cols] = False
        assert np.allclose(np.abs(X[:, t, mask]), 0)


def test_ofdm_noiseless_observation(ofdm):
    """无噪时每个子载波的观测 = 活跃天线的频域信道 × 导频符号"""
    np.random.seed(0)
    H = ofdm.generate_channel(16)
    X = ofdm.generate_pilot_symbols(16)
    Y, _ = ofdm.generate_received_signal(H, X, 120.0)
    err = np.abs(Y - np.einsum('nrkt,ntk->nrk', H, X)).max()
    assert err < 1e-4        # 120 dB 下仅剩有限精度残差、无结构性误差


def test_ofdm_freq_correlation_matches_pdp(ofdm):
    """子载波相关经验值 ≈ PDP 解析式 F[Δ] = Σ_l pdp_l·exp(-2πjΔl/N)"""
    np.random.seed(3)
    H = ofdm.generate_channel(2000)
    for lag in (1, 2, 5):
        r_emp = np.mean(np.conj(H[:, :, :-lag, :]) * H[:, :, lag:, :])
        r_ana = (ofdm.pdp * np.exp(-2j * np.pi * lag * np.arange(4) / 16)).sum()
        assert abs(r_emp - r_ana) < 0.05


def test_ofdm_config_validation():
    with pytest.raises(ValueError, match='整数倍'):
        OFDMDataGenerator(3, 4, 16)
    with pytest.raises(ValueError, match='抽头'):
        OFDMDataGenerator(4, 4, 16, n_taps=20)


# ------------------------------------------------------------- OFDM 估计器

def test_comb_ls_exact_at_own_pilot_grid(ofdm):
    """无噪时：每根天线在自己占据的栅格点上 LS 精确复原信道"""
    np.random.seed(5)
    H = ofdm.generate_channel(8)
    X = ofdm.generate_pilot_symbols(8)
    Y, _ = ofdm.generate_received_signal(H, X, 120.0)
    ls = OFDMCombLSEstimator().estimate(Y, X)
    for t in range(4):
        cols = np.arange(t, 16, 4)
        np.testing.assert_allclose(ls[:, :, cols, t], H[:, :, cols, t], atol=1e-5)


def test_mmse_interp_beats_linear_interp(ofdm):
    """15 dB 下 MMSE 频域插值应显著优于 comb-LS 线性插值"""
    np.random.seed(1)
    H, X, Y, noise_var = ofdm.generate(512, 15.0)
    ls_nmse = calculate_performance_metrics(H, OFDMCombLSEstimator().estimate(Y, X))['nmse']
    mmse_nmse = calculate_performance_metrics(
        H, OFDMLMMSEEstimator(noise_var, ofdm.n_taps, ofdm.pdp).estimate(Y, X))['nmse']
    assert mmse_nmse < 0.5 * ls_nmse


def test_mmse_interp_converges_with_prior(ofdm):
    """噪声趋零且 PDP 先验正确时，MMSE 频域插值逼近真实信道（NMSE → 0）"""
    np.random.seed(2)
    H = ofdm.generate_channel(256)
    X = ofdm.generate_pilot_symbols(256)
    Y, _ = ofdm.generate_received_signal(H, X, 60.0)
    ppm = OFDMLMMSEEstimator(1e-6, ofdm.n_taps, ofdm.pdp).estimate(Y, X)
    assert (np.abs(H - ppm) ** 2).mean() < 0.05


# ------------------------------------------------------------ 特征层 4 维

def test_4d_complex_real_roundtrip():
    rng = np.random.default_rng(7)
    H = (rng.normal(size=(6, 4, 16, 4)) + 1j * rng.normal(size=(6, 4, 16, 4))
         ).astype(np.complex64)
    target = complex_to_real_tensor(H)
    assert target.shape == (6, 4, 2 * 16 * 4)
    pred = target  # 同一 [re|im] 拼接布局逆变换
    H_back = to_complex(pred, n_tx=4, n_sc=16)
    np.testing.assert_allclose(H_back, H, atol=1e-5)


def test_ofdm_input_mode_sizes():
    n_tx, n_rx, n_sc = 4, 4, 16
    assert get_input_size('raw', n_rx, n_tx, n_sc) == 2 * n_sc * (n_rx + n_tx)
    assert get_input_size('ls_refine', n_rx, n_tx, n_sc, n_sc=n_sc) == 2 * n_rx * n_sc * n_tx
    assert (get_input_size('ls_refine_x', n_rx, n_tx, n_sc, n_sc=n_sc)
            == 2 * n_rx * n_sc * n_tx + 2 * n_tx * n_sc)


def test_build_dl_inputs_ofdm(ofdm):
    np.random.seed(9)
    H, X, Y, _ = ofdm.generate(24, 10.0)
    for mode in ('raw', 'ls_refine', 'ls_refine_x'):
        feats = build_dl_inputs(H, X, Y, mode)
        assert feats.shape[1] == get_input_size(mode, 4, 4, 16, n_sc=16)
        assert feats.shape[0] == 24
        assert np.isfinite(feats).all()


# --------------------------------------------------------- LmmseNetAtt 模型

def test_lmmse_net_att_trainable():
    """两步 SGD：从纯物理层起步后损失下降、注意力与第二段物理层梯度流通"""
    torch = pytest.importorskip('torch')
    from deep_learning.models import LmmseNetAttEstimator
    n_rx, n_tx, n_pilot = 4, 4, 16
    input_size = 2 * n_rx * n_tx + 2 * n_tx * n_pilot
    torch.manual_seed(0)
    model = LmmseNetAttEstimator(input_size=input_size, n_rx=n_rx, n_tx=n_tx,
                                 n_pilot=n_pilot)
    x = torch.randn(32, input_size)
    target = torch.randn(32, n_rx, 2 * n_tx)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    # 第 1 步：校正头零初始化 ⇒ 起步输出即第一段物理层（只有头与物理层先验有梯度）
    loss0 = (model(x) - target).pow(2).mean()
    opt.zero_grad()
    loss0.backward()
    opt.step()

    # 第 2 步：头权重已非零，注意力 / 二段物理层路径梯度可流通，损失应下降
    out = model(x)
    loss1 = (out - target).pow(2).mean()
    loss1.backward()
    assert loss1.item() < loss0.item()
    assert model.tok_proj.weight.grad.abs().sum() > 0
    assert model.stage2_noise.grad.abs().sum() > 0


def test_lmmse_net_att_rejects_bad_input():
    torch = pytest.importorskip('torch')
    from deep_learning.models import LmmseNetAttEstimator
    with pytest.raises(ValueError, match='ls_refine_x'):
        LmmseNetAttEstimator(input_size=999, n_rx=4, n_tx=4, n_pilot=16)


def test_lmmse_net_att_flat_only():
    torch = pytest.importorskip('torch')
    from deep_learning.models import LmmseNetAttEstimator
    with pytest.raises(ValueError, match='n_sc'):
        LmmseNetAttEstimator(input_size=160, n_rx=4, n_tx=4, n_pilot=16, n_sc=2)


# ------------------------------------------------------------ OFDM BER 链路

def test_ofdm_ber_genie_and_failure(ofdm):
    np.random.seed(11)
    H, X, Y, noise_var = ofdm.generate(64, 30.0)
    rng = np.random.default_rng(11)
    ber_genie = simulate_qpsk_ber_ofdm(H, H, 30.0, n_symbols=20, rng=rng)
    assert ber_genie < 0.02
    broken = np.zeros_like(H)
    ber_broken = simulate_qpsk_ber_ofdm(H, broken, 30.0, n_symbols=20, rng=rng)
    assert 0.4 < ber_broken <= 0.6


def test_ofdm_ber_requires_enough_rx():
    H = np.zeros((2, 2, 4, 4), dtype=complex)
    with pytest.raises(ValueError, match='ZF'):
        simulate_qpsk_ber_ofdm(H, H, 10.0, n_symbols=4, equalizer='zf')
