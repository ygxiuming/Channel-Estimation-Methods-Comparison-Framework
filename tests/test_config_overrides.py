"""
--set 命令行配置覆盖测试

回归背景：apply_set_overrides 曾在遍历点号路径时丢弃 setdefault 的返回值，
导致 walk 永远停留在根节点，所有嵌套键的覆盖被静默写入顶层（无效覆盖）。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import apply_set_overrides, load_config


@pytest.fixture
def cfg():
    return load_config(str(Path(__file__).resolve().parent.parent
                           / 'src' / 'config' / 'ofdm_freq_selective.yml'))


def test_override_nested_scalar(cfg):
    apply_set_overrides(cfg, ['channel.n_samples=768', 'training.epochs=2'])
    assert cfg['channel']['n_samples'] == 768
    assert cfg['training']['epochs'] == 2


def test_override_deep_path(cfg):
    apply_set_overrides(cfg, ['traditional.lmmse.noise_var_estimation=blind'])
    assert cfg['traditional']['lmmse']['noise_var_estimation'] == 'blind'


def test_override_value_type_parsing(cfg):
    apply_set_overrides(cfg, ['channel.snr_list=[0, 20]',
                              'experiment.seed=123'])
    assert cfg['channel']['snr_list'] == [0, 20]
    assert cfg['experiment']['seed'] == 123


def test_override_missing_mid_node_creates(cfg):
    apply_set_overrides(cfg, ['new_section.new_key=1'])
    assert cfg['new_section']['new_key'] == 1


def test_override_bad_format_raises():
    with pytest.raises(ValueError, match='KEY=VALUE'):
        apply_set_overrides({}, ['nonsense'])
