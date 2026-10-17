#!/usr/bin/env python
"""训练竞速动图渲染器

从传统训练器导出的 results.csv（epoch, train_loss, val_loss, val_nmse_db, lr）
渲染逐 epoch 推进动画，直观对比各 DL 组合验证 NMSE(dB) 的收敛速度与最终水平。

用法:
    uv run python scripts/render_training_race.py \
        --pattern 'results/ab_input_modes_*/results.csv' \
        --out plots/animations/training_race.gif

也可用作任意实验族的复用工具：换 --pattern 即可。
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from utils.plotting import setup_chinese_font  # noqa: E402

MODEL_LABEL = {'cnn': 'CNN', 'rnn': 'RNN', 'lstm': 'LSTM', 'gru': 'GRU',
               'hybrid': 'Hybrid', 'lmmsenet': 'LmmseNet'}
MODE_LABEL = {'raw': 'raw', 'raw_sym': 'raw_sym', 'ls_refine': 'ls_refine',
              'ls_refine_x': 'ls_refine_x'}
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
          '#9467bd', '#17becf', '#8c564b', '#e377c2']


def parse_label(dir_name: str) -> str:
    """从 ab_input_modes_gru_raw_sym 这类目录名提取 'GRU · raw_sym'"""
    m = re.match(r'.*?_([a-z]+?)_((?:raw|ls_refine)(?:_sym|_x)?)$', dir_name)
    if not m:
        return dir_name
    model, mode = MODEL_LABEL.get(m.group(1), m.group(1)), MODE_LABEL.get(m.group(2), m.group(2))
    return f'{model} · {mode}'


def load_series(csv_path: Path) -> list[float]:
    vals = []
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader((ln for ln in f if not ln.lstrip().startswith('#'))):
            try:
                vals.append(float(row['val_nmse_db']))
            except (KeyError, ValueError):
                break
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description='渲染验证 NMSE(dB) 训练竞速 GIF')
    ap.add_argument('--pattern', default='results/ab_input_modes_*/results.csv')
    ap.add_argument('--out', default='plots/animations/training_race.gif')
    ap.add_argument('--frames', type=int, default=90)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--dpi', type=int, default=92)
    args = ap.parse_args()

    series = []
    for csv_path in sorted(Path().glob(args.pattern)):
        vals = load_series(csv_path)
        if len(vals) >= 2:
            series.append((parse_label(csv_path.parent.name), vals))
    if not series:
        raise SystemExit(f'未在 {args.pattern} 找到可用的 results.csv')

    setup_chinese_font()

    max_epochs = max(len(v) for _, v in series)
    # 下界参考：8 条曲线与 LMMSE 水平线们的取值范围
    lo = min(min(v) for _, v in series) - 2
    hi = max(max(v) for _, v in series) + 1.5
    lo = max(lo, -26.0)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    lines = [ax.plot([], [], lw=1.8, color=c, label=name)[0]
             for (name, _), c in zip(series, COLORS)]
    best_text = ax.text(0.985, 0.965, '', transform=ax.transAxes, ha='right',
                        va='top', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.32', fc='#fffbe6', ec='#c9b458', alpha=0.9))

    def frame(f: int):
        upto = max(1, round((f + 1) * max_epochs / args.frames))
        best_name, best_val = None, float('inf')
        for line, (name, vals) in zip(lines, series):
            seg = vals[:upto]
            line.set_data(range(1, len(seg) + 1), seg)
            if seg and seg[-1] < best_val:
                best_name, best_val = name, seg[-1]
        ax.set_xlim(1, max_epochs)
        ax.set_ylim(lo, hi)
        best_text.set_text(f'当前最佳: {best_name}  {best_val:.2f} dB')
        return lines + [best_text]

    ax.set_xlabel('Epoch')
    ax.set_ylabel('验证集 NMSE (dB)')
    ax.set_title('DL 估值器训练竞速：各输入模式 × 模型的验证 NMSE 收敛过程', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.subplots_adjust(right=0.78)

    # 逐帧渲染 + PIL 拼 GIF（避免各平台 MovieWriter 后端差异）
    import numpy as np
    from PIL import Image

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in range(args.frames):
        frame(f)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(rgba.copy()))
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=1000 / args.fps, loop=0, optimize=True)
    plt.close(fig)
    print(f'GIF 已生成: {out}（{args.frames} 帧，覆盖 {len(series)} 条曲线，'
          f'{out.stat().st_size / 1e6:.1f} MB）')


if __name__ == '__main__':
    main()
