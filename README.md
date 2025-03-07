# 信道估计方法比较研究

本项目实现了一个完整的信道估计方法比较研究框架，包括传统方法和深度学习方法的实现、评估和性能对比。

## 项目概述

本项目旨在比较不同信道估计方法的性能，包括：

### 传统方法
- LS（最小二乘）估计
- LMMSE（线性最小均方误差）估计
- ML（最大似然）估计

### 深度学习方法
- CNN（卷积神经网络）
- RNN（循环神经网络）
- LSTM（长短期记忆网络）
- GRU（门控循环单元）
- Hybrid（混合CNN-LSTM模型）

## 项目结构

```
project/
├── src/                    # 源代码目录
│   ├── main.py            # 主程序入口
│   ├── config/            # 配置文件目录
│   │   └── default.yml    # 默认配置文件
│   ├── traditional/       # 传统方法实现
│   │   └── estimators.py  # 传统估计器实现
│   ├── deep_learning/     # 深度学习方法实现
│   │   └── models.py      # 深度学习模型实现
│   └── utils/             # 工具函数
│       ├── config.py      # 配置加载器
│       ├── data_generator.py  # 数据生成器
│       ├── preprocessing.py   # 数据预处理
│       └── trainer.py     # 模型训练器
├── results/               # 实验结果目录
│   ├── logs/             # 日志文件
│   ├── models/           # 保存的模型
│   └── plots/            # 结果图表
└── README.md             # 项目说明文档
```

## 功能特性

1. **数据处理**
   - 支持多种信道模型（Rayleigh、Rician等）
   - 灵活的数据生成和预处理
   - 数据增强支持
   - 自动数据集划分（训练/验证/测试）

2. **模型实现**
   - 传统方法的矩阵运算优化
   - 多种深度学习架构
   - GPU加速支持
   - 模型保存和加载

3. **训练与评估**
   - 自动化训练流程
   - 早停机制
   - 学习率自适应调整
   - 多指标性能评估

4. **可视化与分析**
   - 训练过程可视化
   - 性能对比图表
   - 详细的日志记录
   - 结果导出和保存

## 配置说明

### 实验配置
```yaml
experiment:
  name: "channel_estimation"          # 实验名称
  seed: 42                           # 随机种子
  save_dir: "results/models"         # 模型保存目录
  plot_dir: "results/plots"          # 图表保存目录
  use_cuda: true                     # 是否使用GPU

channel:
  n_tx: 4                           # 发射天线数
  n_rx: 4                           # 接收天线数
  n_pilot: 16                       # 导频长度
  snr_db: 10                        # 信噪比(dB)
  n_samples: 10000                  # 样本数量
  type: "rayleigh"                  # 信道类型
  rician_k: 1.0                     # Rician K因子（仅Rician信道）

data:
  preprocessing:
    normalization: "standard"       # 标准化方法
    remove_outliers: true          # 是否移除异常值
    outlier_threshold: 3.0         # 异常值阈值
  
  augmentation:
    enabled: true                  # 是否启用数据增强
    methods: ["noise", "phase"]    # 增强方法
    noise_std: 0.01               # 噪声标准差
    phase_shift_range: [-0.1, 0.1] # 相位偏移范围
  
  split:
    train_ratio: 0.7              # 训练集比例
    val_ratio: 0.15               # 验证集比例
  
  loader:
    batch_size: 128               # 批次大小
    num_workers: 4                # 数据加载线程数
    pin_memory: true              # 是否固定内存

models:
  common:
    learning_rate: 0.001          # 学习率
  
  cnn:
    channels: [64, 128, 256]      # CNN通道数
    kernel_size: 3                # 卷积核大小
    
  rnn:
    hidden_size: 256              # 隐藏层大小
    num_layers: 2                 # 层数
    
  lstm:
    hidden_size: 256
    num_layers: 2
    
  gru:
    hidden_size: 256
    num_layers: 2
    
  hybrid:
    rnn_hidden_size: 256         # RNN部分隐藏层大小
    cnn_channels: [64, 128]      # CNN部分通道数

training:
  epochs: 100                    # 训练轮数
  early_stopping:
    patience: 10                 # 早停耐心值
```

### 传统方法参数
```yaml
traditional:
  ls:
    regularization: true         # 是否使用正则化
    lambda: 0.01                # 正则化参数
  
  lmmse:
    adaptive_snr: true          # 是否自适应SNR
    correlation_method: "empirical"  # 相关矩阵计算方法
  
  ml:
    max_iter: 100               # 最大迭代次数
    tol: 1e-6                  # 收敛阈值
    learning_rate: 0.01        # 学习率
```

## 使用说明

1. **环境配置**
   ```bash
   # 创建虚拟环境
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   .\venv\Scripts\activate  # Windows
   
   # 安装依赖
   pip install -r requirements.txt
   ```

2. **运行实验**
   ```bash
   # 使用默认配置
   python src/main.py
   
   # 指定配置文件
   python src/main.py --config path/to/config.yml
   
   # 指定设备
   python src/main.py --device cuda
   
   # 设置随机种子
   python src/main.py --seed 42
   ```

3. **查看结果**
   - 实验日志：`results/logs/`
   - 训练好的模型：`results/models/`
   - 性能对比图表：`results/plots/`
   - 实验结果JSON：`results/results_*.json`

## 性能指标

项目使用以下指标评估估计器性能：

1. **MSE (均方误差)**
   - 衡量估计值与真实值的平均平方差
   - 越小越好

2. **NMSE (归一化均方误差)**
   - 考虑信道功率归一化后的MSE
   - 消除信道功率差异的影响

3. **BER (误比特率)**
   - 在给定信道估计下的通信系统误比特率
   - 实际通信性能的直接指标

## 实验结果

实验会生成以下可视化结果：

1. **整体性能对比**
   - 所有方法的MSE柱状图对比
   - 直观展示各方法的估计精度

2. **训练过程分析**
   - 每个深度学习模型的训练损失曲线
   - 学习率变化曲线
   - 帮助理解模型训练动态

3. **多维度性能对比**
   - 传统方法和深度学习方法的雷达图
   - 从多个指标综合评估性能

4. **误差分布分析**
   - 各方法MSE的箱线图
   - 展示估计误差的统计特性

## 注意事项

1. **硬件要求**
   - 建议使用GPU进行训练
   - 至少8GB内存
   - 存储空间：约1GB（取决于实验规模）

2. **数据处理**
   - 注意数据预处理的标准化方法选择
   - 合理设置异常值阈值
   - 根据实际需求调整数据增强参数

3. **训练优化**
   - 适当调整批次大小和学习率
   - 注意早停参数的设置
   - 监控训练过程避免过拟合

4. **结果分析**
   - 综合考虑多个性能指标
   - 注意不同信道条件下的表现
   - 考虑计算复杂度和实时性要求

# 作者信息

作者：修明
邮箱：lzmpt@qq.com

