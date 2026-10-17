<div align="center">

# Channel-Estimation-Methods-Comparison-Framework

**同一个数据管线、同一套指标口径下的传统与深度学习信道估计对比框架**

[![CI](https://github.com/ygxiuming/Channel-Estimation-Methods-Comparison-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/ygxiuming/Channel-Estimation-Methods-Comparison-Framework/actions)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](pyproject.toml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

简体中文 ｜ [English](README.en.md)

</div>

---

深度学习信道估计的论文常常在"各自的实验口径"下与传统方法比较：数据集不同、指标实现不同、训练协议不同，结论便难以复现。这个仓库把全部方法放进**同一套管线**——相同的 Rayleigh 信道样本、相同的导频结构、相同的三次重复评估与置信区间，LS / LMMSE / ML 与 CNN / RNN / LSTM / GRU / Hybrid / LmmseNet 在完全一致的条件里同场竞技。

跑一次完整实验约数分钟，产出 NMSE/BER 曲线、热力图与一份自动撰写的对比报告。当前默认结果：合成高斯世界下 LMMSE 即贝叶斯最优（线性估计已是全局最优，DL 的诚实目标是"不掌握解析先验也能逼近它"）。经过两轮输入模式修复、LmmseNet unfolding、正交导频与 3 倍训练池的接力推进，DL 与 LMMSE 的差距从 **23.9 dB 收窄到 10.6 dB**；在 OFDM 频率选择性信道上，DL 首次稳定超越无先验实用基线（comb-LS 线性插值）**+2.7~5.1 dB**。

![NMSE 对比](plots/nmse_vs_snr_20261017_200009.png)

## 主要研究发现

以下数字均来自仓库内保存的实验产物，可在[文末产物索引](#实验产物索引)中找到对应的完整报告。

**1. 端到端 `raw` 输入让时序模型学不到东西，按导频符号重排后恢复（A/B 实测）**

拼接式的 `raw` 输入破坏了 RNN 感受野的局部性——GRU 训完 NMSE 仍接近 1.0，等于没学。改成按导频符号分组的 `raw_sym` 排布后，GRU@30 dB 的 NMSE 从 `9.94e-01` 降到 `1.79e-01`，NMSE(dB) 提升 **7.4 dB**；`raw` 系曲线在竞速图中 11~12 epoch 即被早停淘汰，而 `raw_sym` 正常收敛：

![训练竞速](plots/animations/training_race.gif)

**2. `ls_refine` 模式缺少导频信息，CNN 在高 SNR 反而把好的初始估计改坏了**

LS 估计在 30 dB 已达 -34.9 dB，但只喂"LS 估计"特征的 CNN 把它 refine 成了 -11.1 dB——网络在没有任何先验可对照的情况下瞎改。补上导频实虚部（`ls_refine_x`）后恢复到 **-17.7 dB（+6.5 dB）**。

**3. LmmseNet：模型驱动 unfolding 把 DL 与 LMMSE 的差距收窄近半**

`LmmseNet` 在网络前向中显式执行逐样本 LMMSE 复数滤波（`R·(R + r·(XXᴴ)⁻¹)⁻¹`，`R`、`r` 是可学习的全局标量，其中 `r` 是等效噪声方差），校正头从零初始化、只学物理层没覆盖的残差。物理层保证了"起点即 LMMSE"，因此不存在常规模型那种学废的风险。与 GRU 同条件对照：30 dB 处 NMSE `4.59e-03` vs GRU `1.68e-02`（**+5.7 dB**），与 LMMSE 差距收窄至 **11.7 dB**；5~30 dB 全中高 SNR 段优于 GRU，低 SNR 持平。

| SNR | LmmseNet | GRU(ls_refine_x) | LMMSE（基准） |
|:---:|:---:|:---:|:---:|
| 0 dB | 2.75e-01 | 2.57e-01 | **2.46e-01** |
| 15 dB | **1.31e-02** | 2.47e-02 | 9.76e-03 |
| 30 dB | **4.59e-03** | 1.68e-02 | 3.13e-04 |

**4. 正交导频（DFT）+ 3 倍训练池：把"学导频统计"变成"学信道插值"（+2.0 dB，解析噪声底也受益）**

随机导频下 `X·Xᴴ` 逐样本随机，逆 Wishart 效应让 LS/LMMSE 的噪声底膨胀约 `n_pilot/(n_pilot−n_tx−1) ≈ 1.45` 倍，也让网络多学一层"导频统计"的随机性。DFT 正交汇把 `X·Xᴴ` 钉成确定性的 `(n_pilot/n_tx)·I`——同一协议（epochs 100 / patience 10，30 dB NMSE）：

| | 随机导频 · 1 万池（A/B） | 随机导频 · 3 万池 | **DFT 正交导频 · 3 万池** |
|:---|:---:|:---:|:---:|
| LmmseNet(ls_refine_x) | 4.58e-03 | 4.04e-03 | **2.89e-03** |
| 与 LMMSE 的差距 | 11.7 dB | 11.1 dB | **10.6 dB** |

三点可分解：纯扩池 **+0.6 dB**、正交导频 **+1.5 dB**，合计 **+2.0 dB**（连解析 LMMSE 的噪声底都收益 ~1.0 dB：`3.13e-04 → 2.50e-04`）。不依赖导频统计的 `raw_sym` 模式获益大得多：GRU(raw_sym)@30 dB 从 A/B 的 `1.79e-01`（−7.5 dB）直落到 **`3.51e-03`（−24.6 dB，+17.1 dB / 51×）**。补充对照（随机导频 · 3 万池的 CNN/GRU `raw_sym` 组，[REPORT_20261017_221652.md](results/REPORT_20261017_221652.md)）把这笔账拆开：随机池扩到 3 倍只值 **+6.2 dB**（`1.79e-01 → 4.3e-02`），剩下 **+10.9 dB** 全部来自正交导频——CNN(raw_sym) 在随机 3 万池下仍是 `6.0e-01`（没救），换 DFT 后直达 `5.9e-03`。[^fn-pilot]

[^fn-pilot]: 同协议验证损失同样悬殊：CNN(raw_sym) 0.317（随机导频）vs 0.0247（DFT 正交导频），≈ +11 dB。

**5. LmmseNetAtt：注意力校正头——一致但小幅的增益**

在 LmmseNet 的一级物理层之上，`LmmseNetAtt` 追加第二段可学习物理层（可学习对角信道先验 `R₂ = diag(softplus(w))` + 二段噪声尺度 `r₂`）与导频符号 token 的 Transformer 校正头，零初始化残差保证"起点即 LmmseNet"。随机导频 · 3 万池、短协议（100 epoch / patience 10）下逐 SNR 点一致小幅领先：

| SNR | LmmseNetAtt | LmmseNet | GRU(ls_refine_x) | LMMSE（基准） |
|:---:|:---:|:---:|:---:|:---:|
| 0 dB | 2.79e-01 | 2.76e-01 | **2.57e-01** | **2.46e-01** |
| 15 dB | **1.23e-02** | 1.27e-02 | 2.24e-02 | 9.76e-03 |
| 30 dB | **3.72e-03** | 4.04e-03 | 1.45e-02 | 3.13e-04 |

30 dB 处与 LMMSE 差距 11.1 → **10.8 dB**（+0.36 dB）。但增益被短协议锁死：校正头零初始化起步即纯物理层，patience=10 使其在第 11 epoch 就被早停，几乎没有展开机会。

**给足耐心后的诚实答案**：改用 300 epoch / patience 30 同协议复跑两种模型（[REPORT_20261017_221016.md](results/REPORT_20261017_221016.md)），Gap 反而只在 ±0.3 dB 内波动——@30 dB LmmseNetAtt `3.97e-03` vs LmmseNet `4.15e-03`（+0.37 dB），且验证损失在初段就封顶（LmmseNetAtt 最优 0.0308 @第 1 轮，LmmseNet 最优 0.0307 @第 2 轮，此后 30 轮耐心内再无改善）。**结论**：在平坦独立 Rayleigh 的合成高斯世界里，LMMSE 本身就是贝叶斯最优，LmmseNet 的物理层起点已经是"内点"，第二段物理层与注意力头没有可学的结构性残差——剩余 ~10.6 dB 的差距不是模型表达力问题，而是"用有限数据超越解析 Bayes 估计"本身的样本效率极限。注意力头的真实用武之地是先验失配 / 非高斯 / 结构化的信道（见路线图）。

**6. OFDM 频选信道：DL 首次把"无先验实用基线"甩开（+2.7~5.1 dB），并给出解析先验的定价**

新的 `channel.type='ofdm_freq_selective'`（16 子载波 / 4 发 × 4 收 / 4 抽头指数 PDP）里，每根天线只占用 1/4 的梳状栅格，估计任务本质是把稀疏观测插值到整个频域网格：

| NMSE | comb-LS 线性插值 | **CNN(ls_refine_x)** | GRU(ls_refine_x) | MMSE 频域插值（PDP 解析先验） |
|:---:|:---:|:---:|:---:|:---:|
| 0 dB | 1.01e+00 | **5.44e-01** | 5.41e-01 | **4.81e-01** |
| 15 dB | 3.40e-01 | **1.19e-01** | 1.26e-01 | 3.05e-02 |
| 30 dB | 3.18e-01 | **9.91e-02** | 1.06e-01 | 9.97e-04 |

三个结构性结论：

- **comb-LS 线性插值被插值损耗"钉死"**：SNR 从 15 → 30 dB，NMSE 仅从 3.4e-01 微降到 3.2e-01（−5 dB 平台）——噪声已经很小，误差全部来自线性插值本身。
- **DL 稳定超越无先验实用基线**：CNN(ls_refine_x) 全 SNR 段领先 comb-LS（0 dB +2.7 dB → 30 dB +5.1 dB），且 0 dB 处几乎逼平 MMSE 插值（仅差 0.5 dB）。
- **解析先验的"定价"**：MMSE 频域插值凭 PDP 先验直达 −30 dB 与 DL 拉开 20 dB——在合成高斯世界，给定正确解析先验的线性估计就是贝叶斯最优，DL 超越的只能是"不掌握先验的实用基线"，这与本文方法论一致。

`raw` 系模式在 comb 结构下失效（GRU(raw) NMSE≈0.95，等于学了个零；CNN(raw) 高 SNR 反超不过 LS）——梳状稀疏结构必须显式喂给网络（`ls_refine_x` 内含该结构），再次印证输入模式决定上限。

![OFDM NMSE 对比](plots/nmse_vs_snr_20261017_220015.png)

## 支持的方法

| 类别 | 方法 | 说明 |
|---|---|---|
| 传统 | LS | 正则有理化的最小二乘（伪逆实现） |
| 传统 | LMMSE | 逐样本滤波，噪声方差支持 oracle / blind（LS 残差盲估计） |
| 传统 | ML | 最大似然（等价于 LS-ZF 口径，作对照） |
| 传统 | **comb-LS / MMSE 频域插值** | OFDM 频选信道基线：梳状点精确 LS + 周期线性插值；MMSE 用 PDP 解析先验做频域维纳插值 |
| 深度学习 | CNN / RNN / LSTM / GRU | 对四个输入模式任意组合训练 |
| 深度学习 | Hybrid | 1D CNN + LSTM 的编解码混合 |
| 深度学习 | **LmmseNetAtt** | LmmseNet + 二段物理层 unfolding（可学习对角先验 + 噪声尺度）+ 导频 token Transformer 校正头，仅用于 `ls_refine_x` 模式 |

深度学习的输入特征有四种：`raw`（逐导频点直拼）、`raw_sym`（按导频符号重排，修复 RNN 局部性）、`ls_refine`（LS 估计实虚部）、`ls_refine_x`（加导频特征）。修复优先的两种已设为默认，`raw`/`ls_refine` 保留以便复现历史实验。OFDM 频选信道下 `raw` 系的导频重排无意义（comb 结构含零），仅 `raw`/`ls_refine_x` 参与且特征层与输出层自动泛化到频域网格。

## 快速开始

需要 Python 3.12 与 [uv](https://docs.astral.sh/uv/)。CUDA 可选，全部功能在 CPU 上也能运行。

```bash
git clone https://github.com/ygxiuming/Channel-Estimation-Methods-Comparison-Framework.git
cd Channel-Estimation-Methods-Comparison-Framework

uv sync                 # 自动装好 Python 3.12 与全部依赖（按 uv.lock 锁定）
uv run python main.py --config src/config/smoke.yml --device cpu   # 约 1 分钟的冒烟验证
uv run python main.py --config src/config/default.yml --device cuda # 完整实验（GPU 数分钟）
uv run pytest tests/ -q  # 60 项测试
```

## CLI

<!-- markdownlint-disable MD013 -->

```text
--config PATH           配置文件（默认 src/config/default.yml）
--device cpu|cuda       覆盖配置里的运行设备
--seed N                覆盖随机种子
--resume                已有 weights/best.pt 的组合跳过训练、直接复用权重（断点续跑）
--set KEY=VALUE         覆盖任意配置项，点号路径，可重复
--replot RESULTS_JSON   不训练不评估，从 results_*.json 重绘全部图表与报告
```

<!-- markdownlint-enable MD013 -->

举例：

```bash
uv run python main.py --set channel.n_tx=8 --set dl.models='[CNN, GRU, LmmseNet]'
uv run python main.py --set traditional.lmmse.noise_var_estimation=blind
uv run python main.py --resume            # 上次训练中断后接着跑；或用旧权重重新评估
uv run python main.py --replot results/results_20261017_200009.json
```

## 实验产物索引

| 产物 | 内容 |
|---|---|
| [results/REPORT_20261017_200009.md](results/REPORT_20261017_200009.md) | 输入模式 A/B 对照（CNN+GRU × 4 模式 × 7 SNR），发现 1、2 的数据来源 |
| [results/REPORT_20261017_202104.md](results/REPORT_20261017_202104.md) | LmmseNet 首验（与 GRU 同条件），发现 3 的数据来源 |
| [results/REPORT_20261017_214249.md](results/REPORT_20261017_214249.md) | 正交导频（DFT）+ 3× 训练池对照（CNN/GRU/LmmseNet × 2 模式），发现 4 的数据来源 |
| [results/REPORT_20261017_215451.md](results/REPORT_20261017_215451.md) | LmmseNetAtt 注意力校正头对照（随机导频 3 万池），发现 5 的数据来源 |
| [results/REPORT_20261017_220015.md](results/REPORT_20261017_220015.md) | MIMO-OFDM 频选信道对比（comb-LS / MMSE 插值 / CNN / GRU），发现 6 的数据来源 |
| [results/REPORT_20261017_221016.md](results/REPORT_20261017_221016.md) | LmmseNet vs LmmseNetAtt 长协议（300 epoch / patience 30）复跑，发现 5 的诚实答案 |
| [results/REPORT_20261017_221652.md](results/REPORT_20261017_221652.md) | 随机导频 · 3 万池 `raw_sym` 对照（CNN/GRU），发现 4 的得益分解 |
| `results/summary_*.csv` / `results/results_*.json` | 各次实验的全量数值（可直接 `--replot`） |
| `plots/*.png`、`plots/animations/*.gif` | 曲线、热力图与训练竞速动图 |

每次运行都会生成独立时间戳目录，互不覆盖；图表与数值随仓库提交，保证论文可复现。

## 配置

全部配置集中在 `src/config/*.yml`（[default.yml](src/config/default.yml) 带逐项注释）。核心字段：

| 字段 | 含义 | 默认 |
|---|---|---|
| `channel.*` | n_tx / n_rx / 导频数 / SNR 列表 / 样本量 | 4×4、16 导频、0~30 dB、1 万样本 |
| `channel.pilot_type` | 导频结构 `random` / `dft` / `hadamard`（正交导频满足 X·Xᴴ=(n_pilot/n_tx)·I） | random |
| `channel.type='ofdm_freq_selective'` | OFDM 频选信道：`n_pilot` 复用为子载波数 N（须为 n_tx 整数倍），`n_taps`/`pdp_decay` 定义归一化指数 PDP | rayleigh |
| `data.*` | 数据增广与 train/val/test 切分 | 关闭增广（保证公平口径） |
| `traditional.lmmse.noise_var_estimation` | `oracle`（真实方差）或 `blind`（LS 残差盲估计） | oracle |
| `dl.input_modes` | 参与的输入模式 | `['raw_sym', 'ls_refine_x']` |
| `dl.models` | 参与的模型 | 全部 6 个（不满足模式要求的组合自动跳过） |
| `training.epochs` / `early_stopping.patience` | 训练协议（所有方法一致） | 100 / 10 |

## 目录结构

```text
├── main.py                     # 主程序入口
├── pyproject.toml / uv.lock    # uv 项目定义与依赖锁定
├── .github/workflows/ci.yml    # GitHub Actions CI（pytest + 冒烟）
├── scripts/
│   ├── release.sh              # Release 自动发布（谷歌双语 notes 模板流程）
│   ├── extract_commits.sh      # conventional commits 提取分组
│   ├── render_training_race.py # 训练竞速 GIF 渲染器
│   └── RELEASE_NOTES_TEMPLATE.md
├── src/
│   ├── config/                 # default / smoke / ab_validation / orthogonal_pilots / attention_unfolding / ofdm_freq_selective
│   ├── traditional/estimators.py    # LS / LMMSE / ML + 盲噪声方差 + OFDM comb-LS / MMSE 频域插值
│   ├── deep_learning/models.py      # CNN/RNN/LSTM/GRU/Hybrid/LmmseNet/LmmseNetAtt
│   └── utils/                  # 数据生成（含 OFDM）、特征构造、训练器、复数域指标、绘图、报告
├── tests/                      # 60 项 pytest 用例（含 LmmseNet 物理层一致性、正交导频/OFDM 链路验证）
├── results/  plots/            # 实验产物（随仓库提交）
└── README.en.md                # English version
```

## 测试与 CI

本地 `uv run pytest tests/ -q`（60 项）覆盖估计器数值正确性、输入特征与盲噪声方差回归、正交导频与 OFDM 链路、`--set` 覆盖回归，以及 LmmseNet 的关键性质——校正头清零并给定真实噪声方差时，输出与解析 LMMSE 逐元素一致。CI 在每次 push / PR 上自动运行同一套测试与 CPU 冒烟实验。

## 版本发布

内置 GitHub CLI 自动发版（参考 Google 产品发布说明的中英双语模板）：

```bash
bash scripts/release.sh --prepare                        # 版本推断 + 生成双语 notes 脚手架
# 按模板润色 release-notes/vX.Y.Z.md
bash scripts/release.sh --notes release-notes/vX.Y.Z.md  # 创建 release 并校验
```

## 路线图

- [x] 四种输入模式与 A/B 实测（v0.2.0）
- [x] LMMSE 盲噪声方差估计（v0.2.0）
- [x] LmmseNet 模型驱动 unfolding（v0.3.0）
- [x] `--resume` / `--set` / `--replot` 工程化 CLI 与 CI（v0.3.0）
- [x] Release 自动化（v0.3.0）
- [x] 正交导频（DFT / Hadamard）与大训练池对照（v1.2.0）
- [x] 注意力 / 多层 unfolding 校正头 LmmseNetAtt（v1.2.0）
- [x] OFDM 频选信道 + comb-LS / MMSE 频域插值基线（v1.2.0）
- [ ] 真实信道数据集（3GPP / 标准 TDL / 实测）验证

## 常见问题

**CUDA 不可用？** 所有方法在 CPU 上可运行，显式加 `--device cpu` 即可。

**图表中文变方块？** `utils/plotting.py` 会自动探测系统中文字体；Ubuntu 下可 `sudo apt install fonts-noto-cjk`。

**为什么 LMMSE 用 oracle 噪声方差？** 上帝信息是基线传统的常见口径；同时也提供 `blind` 模式做无上帝信息的对照，两者可一行配置切换，结果都在产物目录里。

**换信道模型？** `channel.type` 支持 Rayleigh（默认）、Rician 参数 `rician_k` 与 OFDM 频选 `ofdm_freq_selective`（`n_taps` / `pdp_decay` 见配置注释）；`channel.pilot_type` 可叠加任意信道使用。

## 许可证

[MIT](LICENSE) © 修明
