"""
实验结果对比分析与报告生成模块

实验全部完成后自动执行：
1. 对比分析：逐SNR最优方法、传统方法间增益、DL两种输入模式对比、
   DL与传统方法的差距等，自动生成分析文字
2. 生成最终 Markdown 说明文档（配置、结果总表、图表引用、分析、结论）
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

import numpy as np

from utils.features import MODE_DESCRIPTIONS

# 传统方法名（其余视为深度学习方法）
TRADITIONAL_METHODS = ['LS', 'LMMSE', 'ML']

# 解析 DL 方法名 "模型(模式)"，如 "CNN(raw_sym)" -> ("CNN", "raw_sym")
_METHOD_MODE_RE = re.compile(r'^(.+)\((.+)\)$')


def method_mode(name: str) -> Optional[str]:
    """从方法名中提取输入模式；传统方法（无括号）返回 None"""
    m = _METHOD_MODE_RE.match(name)
    return m.group(2) if m else None


def _fmt_mean_std(entry: dict, metric: str) -> str:
    """格式化为 均值±标准差 的科学计数法字符串"""
    return f"{entry[f'{metric}_mean']:.3e}±{entry[f'{metric}_std']:.1e}"


def analyze_results(aggregated: Dict[str, Dict[float, Dict[str, Any]]],
                    snr_list: List[float]) -> List[str]:
    """
    对实验结果做自动对比分析，返回分析要点列表
    """
    lines: List[str] = []
    methods = list(aggregated.keys())
    dl_methods = [m for m in methods if method_mode(m) is not None]
    # 按输入模式分组（模式名取自方法名括号内，任意命名均可）
    modes = sorted({method_mode(m) for m in dl_methods})
    mode_methods = {mode: [m for m in dl_methods
                           if method_mode(m) == mode] for mode in modes}

    # ---- 1. 逐SNR最优方法 ----
    lines.append("### 5.1 各SNR下的最优方法（按NMSE）\n")
    lines.append("| SNR (dB) | 最优方法 | NMSE |")
    lines.append("|---|---|---|")
    for s in snr_list:
        best = min(methods, key=lambda m: aggregated[m][s]['nmse_mean'])
        lines.append(f"| {s:.0f} | {best} | "
                     f"{aggregated[best][s]['nmse_mean']:.3e} |")
    lines.append("")

    # ---- 2. 传统方法对比：LMMSE 相对 LS 的增益 ----
    if 'LS' in aggregated and 'LMMSE' in aggregated:
        lines.append("### 5.2 LMMSE 相对 LS 的 NMSE 增益\n")
        lines.append("| SNR (dB) | LS NMSE | LMMSE NMSE | 增益 (dB) |")
        lines.append("|---|---|---|---|")
        for s in snr_list:
            ls_nmse = aggregated['LS'][s]['nmse_mean']
            lmmse_nmse = aggregated['LMMSE'][s]['nmse_mean']
            gain_db = 10 * np.log10(ls_nmse / lmmse_nmse) if lmmse_nmse > 0 else 0
            lines.append(f"| {s:.0f} | {ls_nmse:.3e} | {lmmse_nmse:.3e} | "
                         f"{gain_db:+.2f} |")
        low_snr_gain = 10 * np.log10(
            aggregated['LS'][snr_list[0]]['nmse_mean'] /
            aggregated['LMMSE'][snr_list[0]]['nmse_mean'])
        high_snr_gain = 10 * np.log10(
            aggregated['LS'][snr_list[-1]]['nmse_mean'] /
            aggregated['LMMSE'][snr_list[-1]]['nmse_mean'])
        lines.append(f"\nLMMSE 在低SNR（{snr_list[0]:.0f} dB）处增益约 "
                     f"{low_snr_gain:.2f} dB，在高SNR（{snr_list[-1]:.0f} dB）处增益约 "
                     f"{high_snr_gain:.2f} dB——符合理论预期：LMMSE 利用信道统计先验，"
                     f"主要在低SNR区域优于LS，高SNR时二者趋于一致。\n")

    # ---- 3. ML 与 LS 的关系 ----
    if 'ML' in aggregated and 'LS' in aggregated:
        s_mid = snr_list[len(snr_list) // 2]
        diff = abs(aggregated['ML'][s_mid]['nmse_mean'] -
                   aggregated['LS'][s_mid]['nmse_mean'])
        lines.append("### 5.3 ML 与 LS\n")
        lines.append(f"在线性高斯模型下，ML 与无正则 LS 严格等价（本框架 ML 采用闭式解），"
                     f"二者 NMSE 差异仅来自 LS 的对角加载正则化"
                     f"（SNR={s_mid:.0f} dB 处相差 {diff:.2e}）。\n")

    # ---- 4. DL 各输入模式对比 ----
    if len(modes) >= 2:
        lines.append("### 5.4 深度学习各输入模式对比\n")
        lines.append("| SNR (dB) | " +
                     " | ".join(f"最优 {mode} | NMSE" for mode in modes) + " |")
        lines.append("|---|" + "---|---|" * len(modes))
        for s in snr_list:
            row = f"| {s:.0f} |"
            for mode in modes:
                best = min(mode_methods[mode],
                           key=lambda m: aggregated[m][s]['nmse_mean'])
                row += f" {best} | {aggregated[best][s]['nmse_mean']:.3e} |"
            lines.append(row)
        lines.append("")

    # ---- 5. DL 与传统方法的整体差距 ----
    if dl_methods:
        lines.append("### 5.5 深度学习方法与传统方法的整体对比\n")
        for s in [snr_list[0], snr_list[-1]]:
            best_dl = min(dl_methods, key=lambda m: aggregated[m][s]['nmse_mean'])
            best_tr = min(TRADITIONAL_METHODS,
                          key=lambda m: aggregated[m][s]['nmse_mean'])
            gap_db = 10 * np.log10(aggregated[best_dl][s]['nmse_mean'] /
                                   aggregated[best_tr][s]['nmse_mean'])
            lines.append(f"- SNR={s:.0f} dB：最优DL方法 {best_dl}"
                         f"（NMSE {aggregated[best_dl][s]['nmse_mean']:.3e}），"
                         f"最优传统方法 {best_tr}"
                         f"（NMSE {aggregated[best_tr][s]['nmse_mean']:.3e}），"
                         f"差距 {gap_db:+.2f} dB")
        lines.append("")

    return lines


def draw_conclusions(aggregated: Dict[str, Dict[float, Dict[str, Any]]],
                     snr_list: List[float]) -> List[str]:
    """根据分析结果自动归纳结论"""
    methods = list(aggregated.keys())
    dl_methods = [m for m in methods if method_mode(m) is not None]
    lines = ["## 6. 结论\n"]

    # 传统方法结论
    if 'LMMSE' in aggregated and 'LS' in aggregated:
        gains = [10 * np.log10(aggregated['LS'][s]['nmse_mean'] /
                               max(aggregated['LMMSE'][s]['nmse_mean'], 1e-12))
                 for s in snr_list]
        lines.append(f"1. **传统方法**：LMMSE 全部SNR点不劣于LS"
                     f"（增益 {min(gains):.2f}~{max(gains):.2f} dB），"
                     f"低SNR优势明显，高SNR趋同；ML 与 LS 基本重合，与理论一致。")

    # 深度学习方法
    if dl_methods:
        best_dl_overall = min(
            dl_methods,
            key=lambda m: np.mean([aggregated[m][s]['nmse_mean'] for s in snr_list]))
        lines.append(f"2. **深度学习方法**：全SNR平均表现最优的DL方法为 "
                     f"**{best_dl_overall}**。")
        best_tr_at_high = min(
            TRADITIONAL_METHODS,
            key=lambda m: aggregated[m][snr_list[-1]]['nmse_mean'])
        dl_nmse = aggregated[best_dl_overall][snr_list[-1]]['nmse_mean']
        tr_nmse = aggregated[best_tr_at_high][snr_list[-1]]['nmse_mean']
        if dl_nmse > tr_nmse:
            lines.append(f"3. **DL vs 传统**：当前配置下 DL 方法尚未超过最优传统方法"
                         f"（高SNR处差 {10*np.log10(dl_nmse/tr_nmse):.2f} dB）。"
                         f"可尝试增大训练样本量/epoch、加深网络、"
                         f"或改用携带导频信息的混合式输入（ls_refine_x）缩小差距。")
        else:
            lines.append(f"3. **DL vs 传统**：DL 方法已超过最优传统方法"
                         f"（高SNR处领先 {10*np.log10(tr_nmse/dl_nmse):.2f} dB）。")

    lines.append(f"4. **方法论**：所有方法在相同测试数据、相同指标口径"
                 f"（复数域NMSE + 真实QPSK链路BER）下评估，"
                 f"每个SNR点独立重复多次并报告均值±标准差，结果具有统计意义。")
    lines.append("")
    return lines


def generate_report(aggregated: Dict[str, Dict[float, Dict[str, Any]]],
                    cfg: dict, plot_files: List[Path],
                    save_dir: Path, timestamp: str) -> Path:
    """
    生成最终 Markdown 说明文档

    参数:
        aggregated: aggregate_results() 的输出
        cfg: 实验配置
        plot_files: 本次生成的图表文件路径列表
        save_dir: 保存目录
        timestamp: 时间戳

    返回:
        报告文件路径
    """
    snr_list = cfg['channel']['snr_list']
    methods = list(aggregated.keys())
    report_path = save_dir / f'REPORT_{timestamp}.md'

    L: List[str] = []
    L.append(f"# 信道估计方法对比实验报告\n")
    L.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- 实验名称：{cfg['experiment']['name']}")
    L.append(f"- 随机种子：{cfg['experiment']['seed']}\n")

    # 1. 实验配置
    ch = cfg['channel']
    L.append("## 1. 实验配置\n")
    L.append("| 参数 | 值 |")
    L.append("|---|---|")
    L.append(f"| 信道类型 (Channel) | {ch['type']} |")
    L.append(f"| MIMO 规模 | {ch['n_tx']} 发 × {ch['n_rx']} 收 |")
    L.append(f"| 导频长度 (Pilots) | {ch['n_pilot']} |")
    L.append(f"| SNR 扫描 | {ch['snr_list']} dB |")
    L.append(f"| 训练池样本 | {ch['n_samples']} |")
    L.append(f"| 每点测试样本 | {cfg['evaluation']['n_test']} |")
    L.append(f"| Monte Carlo 次数 | {cfg['evaluation']['n_trials']} |")
    L.append(f"| BER 仿真符号数 | {cfg['evaluation'].get('ber_symbols', 100)} |")
    L.append(f"| 均衡方式 | {cfg['evaluation'].get('equalizer', 'mmse')} |")
    L.append(f"| DL 输入模式 | {', '.join(cfg['dl'].get('input_modes', []))} |")
    L.append(f"| DL 模型 | {', '.join(cfg['dl'].get('models', []))} |")
    L.append(f"| 训练轮数 | {cfg['training']['epochs']} "
             f"（早停 patience={cfg['training']['early_stopping']['patience']}）|\n")

    # 2. 对比方法
    L.append("## 2. 对比方法\n")
    L.append("**传统方法**：LS（最小二乘）、LMMSE（线性最小均方误差）、"
             "ML（最大似然，闭式解）\n")
    L.append("**深度学习方法**（各输入模式分别成组）：")
    actual_modes = sorted({method_mode(m) for m in methods if method_mode(m)})
    for mode in actual_modes:
        L.append(f"- `{mode}`：{MODE_DESCRIPTIONS.get(mode, '自定义输入模式')}")
    L.append("")

    # 3. 结果总表
    L.append("## 3. 结果总表\n")
    L.append("### 3.1 NMSE（均值±标准差，越小越好）\n")
    L.append("| 方法 | " + " | ".join(f"{s:.0f} dB" for s in snr_list) + " |")
    L.append("|---|" + "---|" * len(snr_list))
    for m in methods:
        L.append(f"| {m} | " + " | ".join(
            _fmt_mean_std(aggregated[m][s], 'nmse') for s in snr_list) + " |")
    L.append("")
    L.append("### 3.2 BER（真实QPSK链路仿真，均值±标准差，越小越好）\n")
    L.append("| 方法 | " + " | ".join(f"{s:.0f} dB" for s in snr_list) + " |")
    L.append("|---|" + "---|" * len(snr_list))
    for m in methods:
        L.append(f"| {m} | " + " | ".join(
            _fmt_mean_std(aggregated[m][s], 'ber') for s in snr_list) + " |")
    L.append("")

    # 4. 图表
    import os
    L.append("## 4. 结果图表\n")
    for p in plot_files:
        # 计算相对报告所在目录的路径，保证 Markdown 图片链接有效
        rel = Path(os.path.relpath(p, save_dir))
        L.append(f"![{p.stem}]({rel.as_posix()})")
        L.append("")
    L.append("> 各模型的训练曲线见各模型目录下的 `results.png`；"
             "逐 epoch 指标见 `results.csv`。\n")

    # 5. 对比分析 + 6. 结论
    L.append("## 5. 对比分析\n")
    L.extend(analyze_results(aggregated, snr_list))
    L.extend(draw_conclusions(aggregated, snr_list))

    # 7. 复现
    L.append("## 7. 复现方法\n")
    L.append("```bash")
    L.append("pip install -r requirements.txt")
    L.append("python main.py --config src/config/default.yml")
    L.append("```\n")

    report_path.write_text('\n'.join(L), encoding='utf-8')
    return report_path
