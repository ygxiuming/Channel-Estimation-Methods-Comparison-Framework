"""
信道估计方法比较研究 —— 主程序（仓库根目录入口）

实验流程：
1. 数据生成：Rayleigh/Rician MIMO 信道 + QPSK 导频 + AWGN（噪声方差可追溯）
2. 深度学习训练（YOLO 风格输出）：训练池混合全部SNR，两种输入模式：
   - 'raw'       : 端到端，输入 [Re(Y),Im(Y),Re(X),Im(X)]
   - 'ls_refine' : 混合式，输入 LS 估计的实虚部，网络学习 refine
   每个模型保存 weights/best.pt 与 weights/last.pt，
   逐 epoch 记录 results.csv，训练结束自动生成 results.png
3. 评估：逐SNR × 逐trial 独立生成测试数据（Monte Carlo），
   传统方法（LS/LMMSE/ML）与DL方法统一在复数域计算 MSE/NMSE，
   BER 通过真实 QPSK 调制-信道-均衡-判决链路仿真获得
4. 输出：NMSE-SNR / BER-SNR 曲线（均值±标准差）、结构化 JSON + CSV 结果

用法:
    python main.py                                   # 默认配置
    python main.py --config src/config/smoke.yml     # 冒烟验证
    python main.py --device cuda --seed 42           # 覆盖设备/种子

作者: lzm lzmpt@qq.com
"""

import sys
from pathlib import Path

# 允许从仓库根目录直接运行：将 src/ 加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import time
import argparse
import random
import csv
from typing import Dict, List, Any
import yaml
import json

from traditional.estimators import (LSEstimator, LMMSEEstimator, MLEstimator,
                                    calculate_performance_metrics,
                                    estimate_noise_var_blind)
from deep_learning.models import (CNNEstimator, RNNEstimator, LSTMEstimator,
                                  GRUEstimator, HybridEstimator)
from utils.data_generator import ChannelDataGenerator
from utils.preprocessing import augment_triplets
from utils.features import (build_dl_inputs, to_complex, complex_to_real_tensor)
from utils.ber import simulate_qpsk_ber
from utils.trainer import ChannelEstimatorTrainer
from utils.plotting import setup_chinese_font, pad_cjk, disp_width
from utils.report import generate_report

# 支持的深度学习模型
MODEL_REGISTRY = {
    'CNN': CNNEstimator,
    'RNN': RNNEstimator,
    'LSTM': LSTMEstimator,
    'GRU': GRUEstimator,
    'Hybrid': HybridEstimator,
}

BANNER = r"""
   ____ _                            _   _____     _   _                 _
  / ___| |__   __ _ _ __  _ __   ___| | | ____|___| |_(_)_ __ ___   __ _| |_ ___  _ __
 | |   | '_ \ / _` | '_ \| '_ \ / _ \ | |  _| / __| __| | '_ ` _ \ / _` | __/ _ \| '__|
 | |___| | | | (_| | | | | | | |  __/ | | |___\__ \ |_| | | | | | | (_| | || (_) | |
  \____|_| |_|\__,_|_| |_|_| |_|\___|_| |_____|___/\__|_|_| |_| |_|\__,_|\__\___/|_|
"""


class TeeLogger:
    """同时将输出写入到控制台和文件的日志记录器"""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def setup_logging(save_dir: Path, timestamp: str) -> TeeLogger:
    """创建日志目录并初始化双向日志记录器"""
    log_dir = save_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return TeeLogger(str(log_dir / f'experiment_{timestamp}.log'))


def set_seed(seed: int):
    """统一设置所有相关库的随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_section_header(title: str):
    """打印带有时间戳的分节标题"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 78}\n {title}  ({timestamp})\n{'=' * 78}")


def print_progress(message: str, indent: int = 0):
    """打印带有时间戳的进度信息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {'  ' * indent}{message}")


def print_config_panel(cfg: dict, device, config_path: str):
    """初始化参数面板（中文为主，英文以()标注）"""
    ch, dl, ev, tr = cfg['channel'], cfg['dl'], cfg['evaluation'], cfg['training']
    rows = [
        ('配置文件 (Config)', config_path),
        ('实验名称 (Experiment)', cfg['experiment']['name']),
        ('随机种子 (Seed)', cfg['experiment']['seed']),
        ('运行设备 (Device)', str(device)),
        ('信道 (Channel)', f"{ch['type']}  {ch['n_tx']}发×{ch['n_rx']}收, "
                           f"{ch['n_pilot']} 导频"),
        ('SNR扫描 (SNR sweep)', f"{ch['snr_list']} dB"),
        ('训练池 (Train pool)', f"{ch['n_samples']} 样本 "
                                f"(训练 {cfg['data']['split']['train_ratio']:.0%} / "
                                f"验证 {cfg['data']['split']['val_ratio']:.0%})"),
        ('评估 (Evaluation)', f"每SNR {ev['n_test']} 测试样本 × "
                              f"{ev['n_trials']} 次重复, "
                              f"BER仿真 {ev.get('ber_symbols', 100)} 符号, "
                              f"{ev.get('equalizer', 'mmse')} 均衡"),
        ('DL输入模式 (DL inputs)', ', '.join(dl.get('input_modes', []))),
        ('DL模型 (DL models)', ', '.join(dl.get('models', []))),
        ('训练 (Training)', f"{tr['epochs']} 轮, "
                            f"批大小 {cfg['data']['loader']['batch_size']}, "
                            f"学习率 {cfg['models']['common']['learning_rate']}, "
                            f"早停耐心 {tr['early_stopping']['patience']}"),
        ('数据增强 (Augmentation)', '启用' if cfg['data'].get('augmentation', {}).get('enabled')
                                  else '关闭'),
        ('LMMSE设置 (LMMSE)', f"{cfg['traditional']['lmmse'].get('correlation_method', 'theoretical')} 协方差, "
                              f"噪声方差 {cfg['traditional']['lmmse'].get('noise_var_estimation', 'oracle')}"),
        ('结果目录 (Save dir)', cfg['experiment']['save_dir']),
        ('图表目录 (Plot dir)', cfg['experiment']['plot_dir']),
    ]
    key_width = max(disp_width(k) for k, _ in rows)
    val_width = max(min(max(disp_width(str(v)) for _, v in rows), 62), 20)
    border = '  +' + '-' * (key_width + 2) + '+' + '-' * (val_width + 2) + '+'
    print(border)
    for key, value in rows:
        value = str(value)
        if disp_width(value) > val_width:
            value = value[:val_width - 3] + '...'
        print(f"  | {pad_cjk(key, key_width)} | {pad_cjk(value, val_width)} |")
    print(border)


def parse_args():
    """解析命令行参数（--config/--device/--seed 均实际生效）"""
    parser = argparse.ArgumentParser(description='信道估计方法比较研究')
    parser.add_argument('--config', type=str, default='src/config/default.yml',
                        help='配置文件路径')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                        help='运行设备（覆盖配置文件）')
    parser.add_argument('--seed', type=int, help='随机种子（覆盖配置文件）')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载并校验YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    required_configs = [
        'experiment.name',
        'channel.n_tx',
        'channel.n_rx',
        'channel.n_pilot',
        'channel.snr_list',
        'channel.n_samples',
        'channel.type',
        'evaluation.n_test',
        'evaluation.n_trials',
    ]
    for path in required_configs:
        if get_config_value(config, path) is None:
            raise ValueError(f"配置文件缺少必要项: {path}")

    return config


def get_config_value(config: dict, path: str) -> Any:
    """使用点号分隔的路径从嵌套字典中获取值"""
    value = config
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def predict_model(model, inputs: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    """分批推理，避免一次性占用过多显存"""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            xb = torch.from_numpy(inputs[i:i + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def prepare_dl_datasets(generator: ChannelDataGenerator,
                        H_pool: np.ndarray, X_pool: np.ndarray,
                        train_idx: np.ndarray, val_idx: np.ndarray,
                        snr_list: List[float], cfg: dict) -> Dict[str, Dict[str, DataLoader]]:
    """
    为每种输入模式构造混合SNR的训练/验证数据集

    信道 H 与导频 X 的分布与SNR无关，只生成一次；
    对每个SNR生成对应的接收信号 Y，合并构成混合SNR训练池
    （测试评估使用独立新生成的数据，见 evaluate_all_methods）。
    """
    aug_cfg = cfg['data'].get('augmentation', {})
    aug_enabled = aug_cfg.get('enabled', False)
    loader_cfg = cfg['data']['loader']
    input_modes = cfg['dl'].get('input_modes', ['raw', 'ls_refine'])

    datasets = {}
    for mode in input_modes:
        train_feats, val_feats, train_targets, val_targets = [], [], [], []

        for snr_db in snr_list:
            Y_pool, _ = generator.generate_received_signal(H_pool, X_pool, snr_db)

            H_tr, X_tr, Y_tr = H_pool[train_idx], X_pool[train_idx], Y_pool[train_idx]
            H_va, X_va, Y_va = H_pool[val_idx], X_pool[val_idx], Y_pool[val_idx]

            # 物理一致的同步增强（仅训练集，逐SNR一致地作用于 (H, X, Y)）
            if aug_enabled:
                H_tr, X_tr, Y_tr = augment_triplets(
                    H_tr, X_tr, Y_tr,
                    methods=aug_cfg.get('methods', ['phase_shift', 'magnitude_scale']),
                    phase_shift_range=tuple(aug_cfg.get('phase_shift_range', [-np.pi, np.pi])),
                    magnitude_scale_range=tuple(aug_cfg.get('magnitude_scale_range', [0.8, 1.25]))
                )

            train_feats.append(build_dl_inputs(H_tr, X_tr, Y_tr, mode))
            val_feats.append(build_dl_inputs(H_va, X_va, Y_va, mode))
            train_targets.append(complex_to_real_tensor(H_tr))
            val_targets.append(complex_to_real_tensor(H_va))

        train_feats = np.concatenate(train_feats, axis=0)
        val_feats = np.concatenate(val_feats, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_feats),
                          torch.from_numpy(train_targets)),
            batch_size=loader_cfg['batch_size'], shuffle=True,
            num_workers=loader_cfg.get('num_workers', 0),
            pin_memory=loader_cfg.get('pin_memory', False))
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_feats),
                          torch.from_numpy(val_targets)),
            batch_size=loader_cfg['batch_size'], shuffle=False,
            num_workers=loader_cfg.get('num_workers', 0),
            pin_memory=loader_cfg.get('pin_memory', False))

        datasets[mode] = {'train': train_loader, 'val': val_loader,
                          'input_size': train_feats.shape[1]}
        print_progress(f"输入模式 '{mode}': 训练 {len(train_feats)} / "
                       f"验证 {len(val_feats)} 样本, "
                       f"输入维度 {train_feats.shape[1]}", 1)

    return datasets


def train_dl_models(datasets: Dict[str, Dict[str, DataLoader]],
                    cfg: dict, device) -> Dict[str, Dict[str, Any]]:
    """训练所有配置的深度学习模型（每个 输入模式 × 模型 组合训练一次）"""
    n_rx, n_tx = cfg['channel']['n_rx'], cfg['channel']['n_tx']
    common_cfg = cfg['models']['common']
    model_names = cfg['dl'].get('models', list(MODEL_REGISTRY.keys()))
    trained = {}
    total = len(datasets) * len(model_names)
    counter = 0

    for mode, data in datasets.items():
        for name in model_names:
            if name not in MODEL_REGISTRY:
                print_progress(f"Warning: unknown model '{name}', skipped.", 1)
                continue

            counter += 1
            full_name = f"{name}({mode})"
            print(f"\n{'─' * 78}")
            print(f"  训练模型 [{counter}/{total}] (Training model): {full_name}")
            print(f"{'─' * 78}")

            kwargs = dict(input_size=data['input_size'], n_rx=n_rx, n_tx=n_tx,
                          dropout=common_cfg.get('dropout', 0.3))
            if name in ('RNN', 'LSTM', 'GRU'):
                kwargs['hidden_size'] = cfg['models'][name.lower()]['hidden_size']
                kwargs['num_layers'] = cfg['models'][name.lower()]['num_layers']
            elif name == 'Hybrid':
                kwargs['hidden_size'] = cfg['models']['hybrid']['rnn_hidden_size']
            model = MODEL_REGISTRY[name](**kwargs)

            trainer = ChannelEstimatorTrainer(
                model=model,
                train_loader=data['train'],
                val_loader=data['val'],
                save_dir=cfg['experiment']['save_dir'],
                project_name=f"{cfg['experiment']['name']}_{name.lower()}_{mode}",
                device=device,
                learning_rate=common_cfg['learning_rate'],
                use_tensorboard=cfg.get('tensorboard', {}).get('enabled', True))

            trainer.train(
                epochs=cfg['training']['epochs'],
                early_stopping_patience=cfg['training']['early_stopping']['patience'])

            trainer.load_model('best.pt')
            trained[full_name] = {'model': trainer.model, 'trainer': trainer,
                                  'mode': mode}

    return trained


def evaluate_all_methods(generator: ChannelDataGenerator,
                         trained_models: Dict[str, Dict[str, Any]],
                         R_h_sample: np.ndarray,
                         cfg: dict, device) -> Dict[str, Dict[float, List[Dict[str, float]]]]:
    """
    逐SNR × 逐trial 的 Monte Carlo 评估

    每次评估独立生成全新测试数据；所有方法共用同一份测试数据，
    指标统一为复数域 MSE/NMSE + 真实QPSK链路仿真的BER。
    每个 SNR 评估完成后打印 YOLO 风格的结果表格。
    """
    snr_list = cfg['channel']['snr_list']
    n_test = cfg['evaluation']['n_test']
    n_trials = cfg['evaluation']['n_trials']
    n_tx = cfg['channel']['n_tx']
    base_seed = cfg['experiment']['seed']
    ber_symbols = cfg['evaluation'].get('ber_symbols', 100)
    equalizer = cfg['evaluation'].get('equalizer', 'mmse')
    lmmse_method = cfg['traditional']['lmmse'].get('correlation_method', 'theoretical')
    noise_var_source = cfg['traditional']['lmmse'].get('noise_var_estimation', 'oracle')
    if noise_var_source not in ('oracle', 'blind'):
        raise ValueError("traditional.lmmse.noise_var_estimation 必须是 'oracle' 或 'blind'")

    results: Dict[str, Dict[float, List[Dict[str, float]]]] = {}

    def record(method: str, snr_db: float, metrics: Dict[str, float]):
        results.setdefault(method, {}).setdefault(snr_db, []).append(metrics)

    for snr_db in snr_list:
        print_section_header(f"评估 SNR = {snr_db} dB (Evaluation)")
        for trial in range(n_trials):
            trial_seed = base_seed + 1000 * (trial + 1) + int(snr_db * 10)
            set_seed(trial_seed)

            H, X, Y, noise_var = generator.generate(n_test, snr_db)

            estimates: Dict[str, np.ndarray] = {}

            # ---- 传统方法 ----
            ls = LSEstimator(
                regularization=cfg['traditional']['ls']['regularization'],
                lambda_=cfg['traditional']['ls']['lambda'])
            estimates['LS'] = ls.estimate(Y, X)

            # LMMSE 噪声方差来源：
            #   'oracle' —— 数据生成器返回的真实值（与历史结果可比）
            #   'blind'  —— 由 LS 残差盲估计，不使用上帝信息，对比更公平
            if noise_var_source == 'blind':
                noise_var_lmmse = estimate_noise_var_blind(Y, X)
            else:
                noise_var_lmmse = noise_var
            lmmse = LMMSEEstimator(
                noise_var=noise_var_lmmse,
                correlation_method=lmmse_method,
                channel_cov=R_h_sample if lmmse_method == 'sample' else None)
            estimates['LMMSE'] = lmmse.estimate(Y, X)

            estimates['ML'] = MLEstimator().estimate(Y, X)

            # ---- 深度学习方法 ----
            mode_inputs: Dict[str, np.ndarray] = {}
            for full_name, pack in trained_models.items():
                mode = pack['mode']
                if mode not in mode_inputs:
                    mode_inputs[mode] = build_dl_inputs(H, X, Y, mode)
                pred = predict_model(pack['model'], mode_inputs[mode], device)
                estimates[full_name] = to_complex(pred, n_tx)

            # ---- 统一指标（复数域 MSE/NMSE + 真实BER，同种子公平对比）----
            for method, H_est in estimates.items():
                metrics = calculate_performance_metrics(H, H_est)
                metrics['ber'] = simulate_qpsk_ber(
                    H, H_est, snr_db, n_symbols=ber_symbols,
                    equalizer=equalizer, rng=np.random.default_rng(trial_seed))
                record(method, snr_db, metrics)

            print_progress(f"第 {trial + 1}/{n_trials} 次重复完成 (Trial done).", 1)

        # 每个 SNR 打印结果表格（中文表头）
        print(f"\n  SNR = {snr_db} dB, {n_trials} 次重复 × {n_test} 测试样本:")
        header = (pad_cjk('方法 (Method)', 20) +
                  pad_cjk('NMSE', 14, '>') +
                  pad_cjk('NMSE(dB)', 12, '>') +
                  pad_cjk('误比特率 (BER)', 16, '>'))
        print(f"  {header}")
        print(f"  {'-' * 62}")
        for method in estimates:
            trials = results[method][snr_db]
            nmse = np.mean([t['nmse'] for t in trials])
            nmse_db = np.mean([t['nmse_db'] for t in trials])
            ber = np.mean([t['ber'] for t in trials])
            print(f"  {pad_cjk(method, 20)}{nmse:>14.4e} "
                  f"{nmse_db:>11.2f} {ber:>15.4e}")
        print()

    return results


def aggregate_results(results) -> Dict[str, Dict[float, Dict[str, Any]]]:
    """对每个 (方法, SNR) 汇总各 trial 的均值与标准差"""
    metric_names = ['mse', 'nmse', 'nmse_db', 'ber']
    aggregated = {}
    for method, snr_dict in results.items():
        aggregated[method] = {}
        for snr_db, trials in snr_dict.items():
            entry = {'trials': trials}
            for m in metric_names:
                values = [t[m] for t in trials]
                entry[f'{m}_mean'] = float(np.mean(values))
                entry[f'{m}_std'] = float(np.std(values))
            aggregated[method][snr_db] = entry
    return aggregated


def plot_results(aggregated, cfg: dict, timestamp: str) -> List[Path]:
    """
    绘制实验结果对比图表（中文标注）：
    1. NMSE-SNR 曲线（semilogy，均值±标准差）
    2. BER-SNR 曲线（semilogy，均值±标准差）
    3. NMSE 方法×SNR 热力图（dB）
    4. BER 方法×SNR 热力图
    （各模型的训练曲线由训练器自动生成 results.png，见各模型目录）

    返回:
        生成的图表文件路径列表
    """
    print_section_header("绘制结果对比图表 (Plotting results)")
    setup_chinese_font()
    plot_dir = Path(cfg['experiment']['plot_dir'])
    plot_dir.mkdir(parents=True, exist_ok=True)

    snr_list = cfg['channel']['snr_list']
    methods = list(aggregated.keys())
    plot_files: List[Path] = []

    # ---- 1/2. NMSE-SNR 与 BER-SNR 曲线 ----
    # 方法数超过默认10色循环，使用 tab20 调色板保证各方法颜色可区分
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, max(len(methods), 20)))
    for metric, ylabel, title, marker in [
        ('nmse', 'NMSE', '各方法 NMSE 随 SNR 变化曲线（均值±标准差）', 'o'),
        ('ber', '误比特率 (BER)', '各方法 BER 随 SNR 变化曲线（真实QPSK链路仿真，均值±标准差）', 's'),
    ]:
        plt.figure(figsize=(10, 6))
        for idx, method in enumerate(methods):
            means = [aggregated[method][s][f'{metric}_mean'] for s in snr_list]
            stds = [aggregated[method][s][f'{metric}_std'] for s in snr_list]
            plt.errorbar(snr_list, means, yerr=stds, marker=marker, capsize=3,
                         color=colors[idx], label=method)
        plt.xlabel('信噪比 SNR (dB)')
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.title(title)
        plt.legend()
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        save_path = plot_dir / f'{metric}_vs_snr_{timestamp}.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        plot_files.append(save_path)
        print_progress(f"{ylabel} 曲线已保存至: {save_path}", 1)

    # ---- 3/4. 方法×SNR 热力图 ----
    for metric, title, to_db in [
        ('nmse', '各方法 NMSE(dB) 热力图（方法 × SNR）', True),
        ('ber', '各方法 BER 热力图（方法 × SNR）', False),
    ]:
        grid = np.array([[aggregated[m][s][f'{metric}_mean'] for s in snr_list]
                         for m in methods])
        if to_db:
            grid_show = 10 * np.log10(np.maximum(grid, 1e-12))
            fmt = lambda v: f'{v:.1f}'
            cbar_label = 'NMSE (dB)'
        else:
            grid_show = grid
            fmt = lambda v: f'{v:.1e}'
            cbar_label = 'BER'

        fig, ax = plt.subplots(figsize=(1.6 * len(snr_list) + 3,
                                        0.6 * len(methods) + 2))
        im = ax.imshow(grid_show, aspect='auto',
                       cmap='viridis_r' if not to_db else 'viridis')
        ax.set_xticks(range(len(snr_list)), [f'{s:.0f}' for s in snr_list])
        ax.set_yticks(range(len(methods)), methods)
        ax.set_xlabel('信噪比 SNR (dB)')
        ax.set_ylabel('方法 (Method)')
        ax.set_title(title)
        for i in range(len(methods)):
            for j in range(len(snr_list)):
                ax.text(j, i, fmt(grid_show[i, j]), ha='center', va='center',
                        fontsize=8,
                        color='white' if grid_show[i, j] < np.median(grid_show)
                        else 'black')
        fig.colorbar(im, ax=ax, label=cbar_label)
        plt.tight_layout()
        save_path = plot_dir / f'{metric}_heatmap_{timestamp}.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        plot_files.append(save_path)
        print_progress(f"{title.split('（')[0]}已保存至: {save_path}", 1)

    return plot_files


def save_results(aggregated, cfg: dict, save_dir: Path, timestamp: str):
    """保存结构化JSON结果与汇总CSV"""
    json_data = {
        'config': cfg,
        'results': {
            method: {str(snr): entry for snr, entry in snr_dict.items()}
            for method, snr_dict in aggregated.items()
        }
    }
    result_file = save_dir / f'results_{timestamp}.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print_progress(f"JSON 结果已保存至: {result_file}", 1)

    csv_file = save_dir / f'summary_{timestamp}.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'snr_db', 'mse_mean', 'mse_std',
                         'nmse_mean', 'nmse_std', 'ber_mean', 'ber_std'])
        for method, snr_dict in aggregated.items():
            for snr, entry in sorted(snr_dict.items()):
                writer.writerow([
                    method, snr,
                    f"{entry['mse_mean']:.6e}", f"{entry['mse_std']:.6e}",
                    f"{entry['nmse_mean']:.6e}", f"{entry['nmse_std']:.6e}",
                    f"{entry['ber_mean']:.6e}", f"{entry['ber_std']:.6e}"])
    print_progress(f"CSV 汇总已保存至: {csv_file}", 1)


def main():
    """主函数"""
    args = parse_args()
    cfg = load_config(args.config)

    # 命令行覆盖配置
    if args.device is not None:
        cfg['experiment']['use_cuda'] = (args.device == 'cuda')
    if args.seed is not None:
        cfg['experiment']['seed'] = args.seed

    # 创建保存目录与日志
    save_dir = Path(cfg['experiment']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(save_dir, timestamp)
    sys.stdout = logger

    try:
        print(BANNER)
        print_section_header("信道估计方法对比实验 — 开始 (Channel Estimation Benchmark)")

        seed = cfg['experiment']['seed']
        set_seed(seed)
        device = torch.device(
            'cuda' if cfg['experiment']['use_cuda'] and torch.cuda.is_available()
            else 'cpu')

        # 初始化参数面板
        print_config_panel(cfg, device, args.config)

        # ========== 1. 生成训练池（H/X与SNR无关，只生成一次）==========
        print_section_header("第 1/4 步 — 生成训练数据池 (Generating training data pool)")
        generator = ChannelDataGenerator(
            n_tx=cfg['channel']['n_tx'],
            n_rx=cfg['channel']['n_rx'],
            n_pilot=cfg['channel']['n_pilot'],
            channel_type=cfg['channel']['type'],
            rician_k=cfg['channel'].get('rician_k', 1.0))

        n_samples = cfg['channel']['n_samples']
        H_pool = generator.generate_channel(n_samples)
        X_pool = generator.generate_pilot_symbols(n_samples)
        print_progress(f"信道矩阵 H: {H_pool.shape}, 导频符号 X: {X_pool.shape}", 1)

        # 训练/验证划分（评估测试使用独立新生成的数据，避免泄漏）
        perm = np.random.permutation(n_samples)
        n_train = int(n_samples * cfg['data']['split']['train_ratio'])
        n_val = int(n_samples * cfg['data']['split']['val_ratio'])
        train_idx, val_idx = perm[:n_train], perm[n_train:n_train + n_val]
        print_progress(f"训练集: {n_train}, 验证集: {n_val}", 1)

        # LMMSE 样本相关矩阵：仅从训练集估计（避免测试集数据泄漏）
        H_train_rows = H_pool[train_idx].reshape(-1, cfg['channel']['n_tx'])
        R_h_sample = (H_train_rows.conj().T @ H_train_rows) / H_train_rows.shape[0]

        # ========== 2. 构造DL数据集 ==========
        print_section_header("第 2/4 步 — 准备深度学习数据集 (Preparing DL datasets)")
        datasets = prepare_dl_datasets(
            generator, H_pool, X_pool, train_idx, val_idx,
            cfg['channel']['snr_list'], cfg)

        # ========== 3. 训练DL模型（YOLO 风格）==========
        print_section_header("第 3/4 步 — 训练深度学习模型 (Training DL models)")
        trained_models = train_dl_models(datasets, cfg, device)

        # ========== 4. Monte Carlo 评估 ==========
        print_section_header("第 4/4 步 — Monte Carlo 评估 (Monte Carlo evaluation)")
        results = evaluate_all_methods(
            generator, trained_models, R_h_sample, cfg, device)

        # ========== 汇总、保存、绘图、生成报告 ==========
        print_section_header("汇总实验结果 (Final summary)")
        aggregated = aggregate_results(results)

        snr_list = cfg['channel']['snr_list']
        print(f"\nNMSE 汇总（各次重复均值）:")
        header = pad_cjk('方法 (Method)', 20) + ''.join(
            pad_cjk(f'{s:.0f}dB', 10, '>') for s in snr_list)
        print(f"  {header}")
        print(f"  {'-' * (20 + 10 * len(snr_list))}")
        for method, snr_dict in aggregated.items():
            row = pad_cjk(method, 20) + ''.join(
                f"{snr_dict[s]['nmse_mean']:>10.1e}" for s in snr_list)
            print(f"  {row}")

        save_results(aggregated, cfg, save_dir, timestamp)
        plot_files = plot_results(aggregated, cfg, timestamp)

        # 生成最终对比分析与说明文档
        print_section_header("生成实验报告 (Generating report)")
        report_path = generate_report(aggregated, cfg, plot_files,
                                      save_dir, timestamp)
        print_progress(f"实验报告已保存至: {report_path}", 1)

        print(f"\n实验完成！所有结果已保存至 (Experiment finished): "
              f"{save_dir.resolve()}")
        print(f"  - 实验报告 (Report): {report_path}")
        print(f"  - 图表目录 (Plots):  {Path(cfg['experiment']['plot_dir']).resolve()}")

    except Exception as e:
        print(f"\n实验出错！错误信息 (Error): {str(e)}")
        raise
    finally:
        sys.stdout = sys.__stdout__
        logger.log_file.close()


if __name__ == "__main__":
    main()
