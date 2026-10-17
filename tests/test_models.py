"""
深度学习模型测试（LmmseNet unfolding 物理层正确性等）

1. LmmseNet 仅接受 'ls_refine_x' 特征维度，错配即报错
2. 前向输出形状 (B, n_rx, 2*n_tx) 且有限
3. 核心物理层正确性：校正头清零 + R=1 + r=真实噪声方差 时，
   输出应严格等于解析 LMMSEEstimator（oracle 噪声方差）
4. 端到端小训练损失下降（可收敛性）
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from deep_learning.models import LmmseNetEstimator
from traditional.estimators import LMMSEEstimator
from utils.data_generator import ChannelDataGenerator
from utils.features import build_dl_inputs


@pytest.fixture
def data():
    gen = ChannelDataGenerator(n_tx=4, n_rx=4, n_pilot=16)
    np.random.seed(0)
    return gen.generate(64, snr_db=10.0)


def _make_model(data, freeze_physics: bool = True):
    """构造 LmmseNet 并（可选）把校正头与物理参数固定为理论值"""
    H, X, Y, noise_var = data
    feats = build_dl_inputs(H, X, Y, 'ls_refine_x')
    model = LmmseNetEstimator(input_size=feats.shape[1], n_rx=4, n_tx=4,
                              n_pilot=16)
    if freeze_physics:
        model.head[-1].weight.data.zero_()
        model.head[-1].bias.data.zero_()
        model.log_gain.data.fill_(0.0)                 # R = 1
        model.log_noise.data.fill_(math.log(noise_var))  # r = 真实噪声方差
    return model, feats


class TestLmmseNet:
    def test_input_size_mismatch_raises(self, data):
        """非 ls_refine_x 特征维度应在构造时报错而非前向崩溃"""
        with pytest.raises(ValueError):
            LmmseNetEstimator(input_size=999, n_rx=4, n_tx=4, n_pilot=16)

    def test_output_shape_and_finite(self, data):
        model, feats = _make_model(data)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(feats))
        assert out.shape == (64, 4, 8)
        assert torch.isfinite(out).all()

    def test_physics_layer_equals_analytic_lmmse(self, data):
        """物理层在 R=1、r=真实噪声方差 时应与解析 LMMSE 逐元素一致"""
        H, X, Y, noise_var = data
        model, feats = _make_model(data)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(feats)).numpy()
        h_phys = out[..., :4] + 1j * out[..., 4:]
        oracle = LMMSEEstimator(noise_var=noise_var,
                                correlation_method='theoretical').estimate(Y, X)
        assert np.allclose(h_phys, oracle, atol=1e-4)

    def test_loss_decreases_with_training(self, data):
        """端到端 Adam 小训练几十步，训练损失应下降"""
        H, X, Y, _ = data
        model, feats = _make_model(data, freeze_physics=False)
        target = torch.from_numpy(
            np.concatenate([H.real, H.imag], axis=-1)).float()
        x = torch.from_numpy(feats)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.MSELoss()

        model.train()
        first_loss, last_loss = None, None
        for step in range(60):
            opt.zero_grad()
            loss = crit(model(x), target)
            loss.backward()
            opt.step()
            if step == 0:
                first_loss = loss.item()
            last_loss = loss.item()
        assert last_loss < first_loss, (first_loss, last_loss)
