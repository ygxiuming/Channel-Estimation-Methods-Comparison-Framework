<div align="center">

# Channel-Estimation-Methods-Comparison-Framework

**Classical and deep-learning channel estimation compared in one pipeline, with one metric implementation**

[![CI](https://github.com/ygxiuming/Channel-Estimation-Methods-Comparison-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/ygxiuming/Channel-Estimation-Methods-Comparison-Framework/actions)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](pyproject.toml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[简体中文](README.md) ｜ English

</div>

---

Deep-learning channel estimation papers are usually compared against classical baselines under each paper's own protocol: different datasets, different metric implementations, different training setups. Results stop being reproducible. This repository puts every method into **one pipeline** — identical Rayleigh channel samples, identical pilot structure, identical three-trial evaluation with confidence intervals — so LS / LMMSE / ML race CNN / RNN / LSTM / GRU / Hybrid / LmmseNet under exactly the same conditions.

One full run takes minutes and produces NMSE/BER curves, heatmaps, and an auto-written comparison report. In the synthetic Gaussian world, LMMSE is Bayes-optimal — a linear estimator is already globally optimal, so the honest goal for deep learning is to approach it **without holding any analytical priors**. After two input-mode fixes, LmmseNet unfolding, orthogonal pilots, and a 3× larger training pool, the DL-to-LMMSE gap has been cut from **23.9 dB to 10.6 dB**; on OFDM frequency-selective channels, deep learning now beats the prior-free practical baseline (comb-LS with linear interpolation) by a consistent **+2.7–5.1 dB**.

![NMSE comparison](plots/nmse_vs_snr_20261017_200009.png)

## Key findings

All numbers below come from artifacts stored in this repository; the full reports are listed in the [artifact index](#artifact-index).

**1. End-to-end `raw` inputs prevent sequence models from learning; per-pilot-symbol layout fixes it (A/B measured)**

The concatenated `raw` layout destroys RNN receptive-field locality — after a full training run, GRU NMSE stays near 1.0, i.e. it learned nothing. Regrouping inputs per pilot symbol (`raw_sym`) drops GRU@30 dB NMSE from `9.94e-01` to `1.79e-01`, a **7.4 dB** improvement in NMSE(dB). In the race plot below, `raw` runs get early-stopped within 11–12 epochs while `raw_sym` converges normally:

![Training race](plots/animations/training_race.gif)

**2. `ls_refine` lacks pilot context, so CNN over-corrects a good LS estimate at high SNR**

LS already reaches -34.9 dB at 30 dB SNR, yet a CNN fed only with the LS estimate drags it down to -11.1 dB — with nothing to anchor on, the network edits blindly. Exposing pilot features (`ls_refine_x`) restores it to **-17.7 dB (+6.5 dB)**.

**3. LmmseNet: model-driven unfolding halves the DL-to-LMMSE gap**

LmmseNet explicitly executes the per-sample LMMSE complex filter inside the forward pass (`R·(R + r·(XXᴴ)⁻¹)⁻¹`, with learnable global scalars `R` and `r`, the latter being an equivalent noise variance), while a correction head — initialized to zeros — learns only the residual the physics layer cannot cover. Because the physics layer makes the starting point exactly LMMSE, the "model learns nothing" failure mode is ruled out. Head-to-head against GRU under identical conditions: NMSE `4.59e-03` vs `1.68e-02` at 30 dB (**+5.7 dB**), shrinking the gap to LMMSE to **11.7 dB**; it beats GRU across 5–30 dB and ties at low SNR.

| SNR | LmmseNet | GRU(ls_refine_x) | LMMSE (reference) |
|:---:|:---:|:---:|:---:|
| 0 dB | 2.75e-01 | 2.57e-01 | **2.46e-01** |
| 15 dB | **1.31e-02** | 2.47e-02 | 9.76e-03 |
| 30 dB | **4.59e-03** | 1.68e-02 | 3.13e-04 |

**4. Orthogonal pilots (DFT) + 3× training pool: stop learning pilot statistics, start learning interpolation (+2.0 dB; even the analytical noise floor benefits)**

With random pilots, `X·Xᴴ` is random per sample: the inverse-Wishart effect inflates the LS/LMMSE noise floor by roughly `n_pilot/(n_pilot−n_tx−1) ≈ 1.45×`, and the network must also learn the pilot statistics on top of the channel. A DFT orthogonal comb pins `X·Xᴴ` to the deterministic `(n_pilot/n_tx)·I`. Same protocol (100 epochs / patience 10), NMSE at 30 dB:

| | Random pilots · 10k pool (A/B) | Random pilots · 30k pool | **DFT orthogonal pilots · 30k pool** |
|:---|:---:|:---:|:---:|
| LmmseNet(ls_refine_x) | 4.58e-03 | 4.04e-03 | **2.89e-03** |
| Gap to LMMSE | 11.7 dB | 11.1 dB | **10.6 dB** |

The three points decompose cleanly: larger pool alone **+0.6 dB**, orthogonal pilots **+1.5 dB**, combined **+2.0 dB** — and even the analytical LMMSE noise floor gains ~1.0 dB (`3.13e-04 → 2.50e-04`). The pilot-statistics-free `raw_sym` mode benefits far more: GRU(raw_sym)@30 dB drops from the A/B `1.79e-01` (−7.5 dB) to **`3.51e-03` (−24.6 dB, +17.1 dB / 51×)**. A supplementary control ([REPORT_20261017_221652.md](results/REPORT_20261017_221652.md): CNN/GRU on `raw_sym` with random pilots at the 30k pool) splits the ledger: tripling the random-pilot pool is worth only **+6.2 dB** (`1.79e-01 → 4.3e-02`), and the remaining **+10.9 dB** comes from the orthogonal pilots themselves — CNN(raw_sym) is still `6.0e-01` (unsalvaged) on the random 30k pool vs `5.9e-03` under DFT.[^fn-pilot]

[^fn-pilot]: Validation losses under the same protocol are just as stark: CNN(raw_sym) 0.317 (random pilots) vs 0.0247 (DFT orthogonal pilots), ≈ +11 dB.

**5. LmmseNetAtt: the attention correction head — small but consistent gains**

On top of LmmseNet's first physics layer, `LmmseNetAtt` adds a second learnable physics layer (learnable diagonal channel prior `R₂ = diag(softplus(w))` plus a second-stage noise scale `r₂`) and a pilot-symbol-token Transformer correction head; zero-initialized residuals guarantee "the starting point is LmmseNet". Random pilots · 30k pool, short protocol (100 epochs / patience 10): a small but uniform lead at every SNR:

| SNR | LmmseNetAtt | LmmseNet | GRU(ls_refine_x) | LMMSE (reference) |
|:---:|:---:|:---:|:---:|:---:|
| 0 dB | 2.79e-01 | 2.76e-01 | **2.57e-01** | **2.46e-01** |
| 15 dB | **1.23e-02** | 1.27e-02 | 2.24e-02 | 9.76e-03 |
| 30 dB | **3.72e-03** | 4.04e-03 | 1.45e-02 | 3.13e-04 |

At 30 dB the gap to LMMSE improves from 11.1 to **10.8 dB** (+0.36 dB). But the short protocol caps it: the zero-initialized head starts as a pure physics layer and patience=10 stops training at epoch 11, leaving it barely unfolded.

**The honest answer with patience granted**: re-running both models under a 300-epoch / patience-30 protocol ([REPORT_20261017_221016.md](results/REPORT_20261017_221016.md)) moves everything by at most ±0.3 dB — at 30 dB LmmseNetAtt `3.97e-03` vs LmmseNet `4.15e-03` (+0.37 dB), and validation losses plateau early (best 0.0308 at epoch 1 for LmmseNetAtt, best 0.0307 at epoch 2 for LmmseNet, with no further improvement inside the 30-epoch patience). **Conclusion**: in the flat, independent Rayleigh synthetic Gaussian world LMMSE itself is Bayes-optimal, so LmmseNet's physics-layer starting point is already an interior point — the second physics layer and the attention head have no structural residual left to learn. The remaining ~10.6 dB gap is not a model-capacity problem; it is the sample-efficiency limit of "beating an analytical Bayes estimator from finite data". The attention head's real use case is prior mismatch / non-Gaussian / structured channels (see the roadmap).

**6. OFDM frequency-selective channel: DL pulls ahead of the prior-free practical baseline (+2.7–5.1 dB) and prices the analytical prior**

The new `channel.type='ofdm_freq_selective'` (16 subcarriers / 4 TX × 4 RX / 4-tap exponential PDP) gives each transmit antenna only a quarter of the comb grid, so the task is to interpolate sparse observations onto the whole frequency-domain grid:

| NMSE | comb-LS linear interp | **CNN(ls_refine_x)** | GRU(ls_refine_x) | MMSE frequency-domain interp (analytical PDP prior) |
|:---:|:---:|:---:|:---:|:---:|
| 0 dB | 1.01e+00 | **5.44e-01** | 5.41e-01 | **4.81e-01** |
| 15 dB | 3.40e-01 | **1.19e-01** | 1.26e-01 | 3.05e-02 |
| 30 dB | 3.18e-01 | **9.91e-02** | 1.06e-01 | 9.97e-04 |

Three structural conclusions:

- **comb-LS linear interpolation is pinned by interpolation loss**: from 15 → 30 dB the NMSE barely moves (3.4e-01 → 3.2e-01, a −5 dB plateau) — noise is already negligible; the error is interpolation itself.
- **DL consistently beats the prior-free practical baseline**: CNN(ls_refine_x) leads comb-LS across the whole SNR sweep (+2.7 dB at 0 dB → +5.1 dB at 30 dB), and nearly ties the MMSE interpolator at 0 dB (0.5 dB behind).
- **The "price" of the analytical prior**: the MMSE frequency-domain interpolator, armed with the PDP prior, reaches −30 dB and holds a 20 dB lead — in the synthetic Gaussian world a linear estimator with the correct analytical prior is Bayes-optimal, so DL can only aim to beat prior-free practical baselines, consistent with the methodology of this repo.

The `raw` modes break down under the comb structure (GRU(raw) NMSE ≈ 0.95, i.e. it learned almost nothing; CNN(raw) cannot pass LS at high SNR) — the comb sparsity must be handed to the network explicitly (`ls_refine_x` embeds it), re-confirming that the input mode caps the ceiling.

![OFDM NMSE comparison](plots/nmse_vs_snr_20261017_220015.png)

## Methods

| Category | Method | Notes |
|---|---|---|
| Classical | LS | Regularized least squares (pseudo-inverse) |
| Classical | LMMSE | Per-sample filtering; noise variance supports oracle / blind (LS-residual based) |
| Classical | ML | Maximum likelihood (LS-ZF equivalent, kept as reference) |
| Classical | **comb-LS / MMSE frequency-domain interpolation** | OFDM frequency-selective baselines: exact LS on the comb grid + periodic linear interpolation; MMSE Wiener interpolation with the analytical PDP prior |
| Deep learning | CNN / RNN / LSTM / GRU | Trainable on any of the four input modes |
| Deep learning | Hybrid | 1D CNN + LSTM encoder-decoder |
| Deep learning | **LmmseNet** | Model-driven unfolding with an embedded LMMSE physics layer; requires the `ls_refine_x` mode |
| Deep learning | **LmmseNetAtt** | LmmseNet + a second-stage physics layer (learnable diagonal prior + noise scale) + pilot-token Transformer correction head; requires the `ls_refine_x` mode |

Four input features are available for the deep models: `raw` (per-pilot concatenation), `raw_sym` (regrouped per pilot symbol, fixing RNN locality), `ls_refine` (real/imag LS estimate), and `ls_refine_x` (plus pilot features). The two improved modes are the defaults; the legacy modes remain configurable for A/B reproduction. Under the OFDM frequency-selective channel the `raw`-family pilot regrouping is meaningless (the comb structure contains zeros); the feature/output layers generalize automatically to the frequency-domain grid.

## Getting started

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). CUDA is optional; everything runs on CPU.

```bash
git clone https://github.com/ygxiuming/Channel-Estimation-Methods-Comparison-Framework.git
cd Channel-Estimation-Methods-Comparison-Framework

uv sync                 # installs Python 3.12 and all dependencies (locked by uv.lock)
uv run python main.py --config src/config/smoke.yml --device cpu    # ~1 min smoke test
uv run python main.py --config src/config/default.yml --device cuda # full experiment
uv run pytest tests/ -q  # 60 tests
```

## CLI

<!-- markdownlint-disable MD013 -->

```text
--config PATH           config file (default src/config/default.yml)
--device cpu|cuda       override the configured device
--seed N                override the random seed
--resume                skip combinations that already have weights/best.pt (resume runs)
--set KEY=VALUE         override any config item, dotted path, repeatable
--replot RESULTS_JSON   regenerate every plot and report from results_*.json, no training
```

<!-- markdownlint-enable MD013 -->

Examples:

```bash
uv run python main.py --set channel.n_tx=8 --set dl.models='[CNN, GRU, LmmseNet]'
uv run python main.py --set traditional.lmmse.noise_var_estimation=blind
uv run python main.py --resume            # continue an interrupted run, or re-evaluate with old weights
uv run python main.py --replot results/results_20261017_200009.json
```

## Artifact index

| Artifact | Contents |
|---|---|
| [results/REPORT_20261017_200009.md](results/REPORT_20261017_200009.md) | Input-mode A/B comparison (CNN+GRU × 4 modes × 7 SNRs); source of findings 1 and 2 |
| [results/REPORT_20261017_202104.md](results/REPORT_20261017_202104.md) | LmmseNet first validation against GRU; source of finding 3 |
| [results/REPORT_20261017_214249.md](results/REPORT_20261017_214249.md) | Orthogonal pilots (DFT) + 3× training pool comparison; source of finding 4 |
| [results/REPORT_20261017_215451.md](results/REPORT_20261017_215451.md) | LmmseNetAtt attention correction head comparison (random pilots, 30k pool); source of finding 5 |
| [results/REPORT_20261017_220015.md](results/REPORT_20261017_220015.md) | MIMO-OFDM frequency-selective channel comparison (comb-LS / MMSE interpolation / CNN / GRU); source of finding 6 |
| [results/REPORT_20261017_221016.md](results/REPORT_20261017_221016.md) | LmmseNet vs LmmseNetAtt under the long protocol (300 epochs / patience 30), the honest answer in finding 5 |
| [results/REPORT_20261017_221652.md](results/REPORT_20261017_221652.md) | Random-pilot 30k-pool `raw_sym` control (CNN/GRU), the gain decomposition in finding 4 |
| `results/summary_*.csv` / `results/results_*.json` | Full numbers of every run (replayable via `--replot`) |
| `plots/*.png`, `plots/animations/*.gif` | Curves, heatmaps, and the training-race animation |

Each run writes its own timestamped directory; results and figures are committed with the repository so experiments stay reproducible.

## Configuration

Everything lives in `src/config/*.yml` ([default.yml](src/config/default.yml) is fully commented). Core fields:

| Field | Meaning | Default |
|---|---|---|
| `channel.*` | n_tx / n_rx / pilots / SNR list / sample count | 4×4, 16 pilots, 0–30 dB, 10k samples |
| `channel.pilot_type` | pilot structure `random` / `dft` / `hadamard` (orthogonal pilots satisfy X·Xᴴ=(n_pilot/n_tx)·I) | random |
| `channel.type='ofdm_freq_selective'` | OFDM frequency-selective channel: `n_pilot` doubles as subcarrier count N (must be a multiple of n_tx); `n_taps`/`pdp_decay` define the normalized exponential PDP | rayleigh |
| `data.*` | augmentation and train/val/test split | augmentation off (fair protocol) |
| `traditional.lmmse.noise_var_estimation` | `oracle` (true variance) or `blind` (LS residual) | oracle |
| `dl.input_modes` | input modes in play | `['raw_sym', 'ls_refine_x']` |
| `dl.models` | models in play | all six (invalid model×mode pairs are skipped automatically) |
| `training.epochs` / `early_stopping.patience` | training protocol, identical for all methods | 100 / 10 |

## Repository layout

```text
├── main.py                     # entry point
├── pyproject.toml / uv.lock    # uv project definition and locked dependencies
├── .github/workflows/ci.yml    # GitHub Actions CI (pytest + smoke)
├── scripts/
│   ├── release.sh              # release automation (Google-style bilingual notes)
│   ├── extract_commits.sh      # conventional-commit extraction and grouping
│   ├── render_training_race.py # training-race GIF renderer
│   └── RELEASE_NOTES_TEMPLATE.md
├── src/
│   ├── config/                 # default / smoke / ab_validation / orthogonal_pilots / attention_unfolding / ofdm_freq_selective
│   ├── traditional/estimators.py    # LS / LMMSE / ML + blind noise-variance + OFDM comb-LS / MMSE frequency-domain interpolation
│   ├── deep_learning/models.py      # CNN/RNN/LSTM/GRU/Hybrid/LmmseNet/LmmseNetAtt
│   └── utils/                  # data generation (incl. OFDM), features, trainer, complex metrics, plots, reports
├── tests/                      # 60 pytest cases, incl. LmmseNet physics-layer equivalence, orthogonal pilots, OFDM chains
├── results/  plots/            # experiment artifacts (committed)
└── README.md                   # 简体中文版
```

## Tests and CI

`uv run pytest tests/ -q` (60 cases) covers estimator numerics, input-feature and blind-noise-variance regression, orthogonal pilots / OFDM chains, the `--set` override path, and the key LmmseNet property: with the correction head zeroed and the true noise variance given, the output matches the analytical LMMSE element-wise. CI runs the same suite plus a CPU smoke experiment on every push / pull request.

## Releases

Release automation is built in (Google-style bilingual notes template):

```bash
bash scripts/release.sh --prepare                        # version inference + notes scaffold
# polish release-notes/vX.Y.Z.md following the template
bash scripts/release.sh --notes release-notes/vX.Y.Z.md  # create and verify the release
```

## Roadmap

- [x] Four input modes with A/B measurements (v0.2.0)
- [x] Blind noise-variance estimation for LMMSE (v0.2.0)
- [x] LmmseNet model-driven unfolding (v0.3.0)
- [x] `--resume` / `--set` / `--replot` CLI engineering and CI (v0.3.0)
- [x] Release automation (v0.3.0)
- [x] Orthogonal pilots (DFT / Hadamard) and larger training pools (v1.2.0)
- [x] Attention / multi-layer unfolding correction head LmmseNetAtt (v1.2.0)
- [x] OFDM frequency-selective channel + comb-LS / MMSE frequency-domain interpolation baselines (v1.2.0)
- [ ] Real-world channel datasets (3GPP / standard TDL / measurement-based validation)

## FAQ

**No CUDA?** Everything runs on CPU — add `--device cpu`.

**Boxes instead of Chinese glyphs in plots?** `utils/plotting.py` auto-detects installed CJK fonts; on Ubuntu, `sudo apt install fonts-noto-cjk`.

**Why oracle noise variance for LMMSE?** It is the common convention for classical baselines. A `blind` variant is one config switch away, and both results are preserved in the artifacts.

**Different channel model?** `channel.type` supports Rayleigh (default), Rician (`rician_k`), and the OFDM frequency-selective `ofdm_freq_selective` (`n_taps` / `pdp_decay` in the config comments); `channel.pilot_type` composes with any of them.

## License

[MIT](LICENSE) © Xiu Ming
