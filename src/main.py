"""
信道估计方法比较研究

本模块是项目的主入口，实现了完整的信道估计实验流程，包括：
1. 数据生成与预处理
2. 传统方法评估（LS、LMMSE、ML）
3. 深度学习方法评估（CNN、RNN、LSTM、GRU、Hybrid）
4. 结果可视化与保存

主要功能：
- 配置管理：支持通过YAML文件配置实验参数
- 数据处理：生成、预处理、增强和划分数据集
- 模型训练：支持多种深度学习模型的训练和评估
- 结果分析：生成多种可视化图表进行性能对比
- 日志记录：详细记录实验过程和结果

作者: lzm lzmpt@qq.com
日期: 2025-03-07
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pathlib import Path
import time
import argparse
import random
from typing import Dict
import yaml
import json
from typing import Any
import warnings
import sys
import logging
from contextlib import redirect_stdout

# 过滤掉特定的警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch.optim.lr_scheduler')

from traditional.estimators import LSEstimator, LMMSEEstimator, MLEstimator, calculate_performance_metrics
from deep_learning.models import CNNEstimator, RNNEstimator, LSTMEstimator, GRUEstimator, HybridEstimator
from utils.data_generator import ChannelDataGenerator
from utils.preprocessing import ChannelPreprocessor, augment_data
from utils.trainer import ChannelEstimatorTrainer
from utils.config import Config

class TeeLogger:
    """
    同时将输出写入到控制台和文件的日志记录器
    
    该类实现了一个双向输出流，可以：
    1. 将所有输出同时发送到终端和日志文件
    2. 实时刷新输出，确保日志及时记录
    3. 支持标准输出流的所有基本操作
    
    参数:
        filename (str): 日志文件的路径
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        """写入消息到终端和文件"""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        """刷新输出缓冲区"""
        self.terminal.flush()
        self.log_file.flush()

def setup_logging(save_dir: Path, timestamp: str):
    """
    设置日志记录系统
    
    创建日志目录并初始化日志记录器，支持：
    1. 创建时间戳命名的日志文件
    2. 同时输出到控制台和文件
    3. 自动创建所需目录结构
    
    参数:
        save_dir (Path): 保存目录的路径
        timestamp (str): 时间戳字符串
    
    返回:
        TeeLogger: 配置好的日志记录器
    """
    log_dir = save_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'experiment_{timestamp}.log'
    return TeeLogger(str(log_file))

def set_seed(seed):
    """
    设置随机种子以确保实验可重复性
    
    统一设置所有相关库的随机种子，包括：
    1. Python random模块
    2. NumPy
    3. PyTorch CPU
    4. PyTorch GPU（如果可用）
    5. CUDA后端（如果可用）
    
    参数:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_section_header(title):
    """
    打印带有时间戳的分节标题
    
    创建醒目的分节标题，包括：
    1. 分隔线
    2. 当前时间戳
    3. 节标题
    
    参数:
        title (str): 节标题文本
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*80}\n{timestamp} - {title}\n{'='*80}")

def print_progress(message, indent=0):
    """
    打印带有时间戳的进度信息
    
    格式化输出进度信息，包括：
    1. 时间戳
    2. 缩进级别
    3. 具体消息
    
    参数:
        message (str): 要打印的消息
        indent (int): 缩进级别（每级2个空格）
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    indent_str = "  " * indent
    print(f"[{timestamp}] {indent_str}{message}")

def evaluate_traditional_methods(H: np.ndarray, X: np.ndarray, Y: np.ndarray, snr: float, cfg: dict) -> Dict[str, Dict[str, float]]:
    """
    评估传统信道估计方法的性能
    
    实现了三种传统估计方法的评估：
    1. LS（最小二乘）估计
    2. LMMSE（线性最小均方误差）估计
    3. ML（最大似然）估计
    
    对每种方法：
    - 初始化估计器
    - 执行信道估计
    - 计算性能指标
    - 记录执行时间
    - 输出详细结果
    
    参数:
        H (np.ndarray): 真实信道矩阵，shape=(n_samples, n_rx, n_tx)
        X (np.ndarray): 导频符号，shape=(n_samples, n_tx, n_pilot)
        Y (np.ndarray): 接收信号，shape=(n_samples, n_rx, n_pilot)
        snr (float): 信噪比（线性，非dB）
        cfg (dict): 配置字典，包含各估计器的参数
    
    返回:
        Dict[str, Dict[str, float]]: 包含各估计器性能指标的嵌套字典
    """
    results = {}
    
    # LS估计
    print_progress("开始 LS 估计...", 1)
    start_time = time.time()
    ls_estimator = LSEstimator(
        regularization=cfg['traditional']['ls']['regularization'],
        lambda_=cfg['traditional']['ls']['lambda']
    )
    H_est_ls = ls_estimator.estimate(Y, X)
    results['ls'] = calculate_performance_metrics(H, H_est_ls)
    print_progress(f"LS 估计完成 (耗时: {time.time()-start_time:.2f}s)", 1)
    print_progress("LS 性能指标:", 2)
    for metric, value in results['ls'].items():
        print_progress(f"{metric}: {value:.6f}", 3)
    
    # LMMSE估计
    print_progress("\n开始 LMMSE 估计...", 1)
    start_time = time.time()
    lmmse_estimator = LMMSEEstimator(
        snr=snr,
        adaptive_snr=cfg['traditional']['lmmse']['adaptive_snr'],
        correlation_method=cfg['traditional']['lmmse']['correlation_method']
    )
    H_est_lmmse = lmmse_estimator.estimate(Y, X)
    results['lmmse'] = calculate_performance_metrics(H, H_est_lmmse)
    print_progress(f"LMMSE 估计完成 (耗时: {time.time()-start_time:.2f}s)", 1)
    print_progress("LMMSE 性能指标:", 2)
    for metric, value in results['lmmse'].items():
        print_progress(f"{metric}: {value:.6f}", 3)
    
    # ML估计
    print_progress("\n开始 ML 估计...", 1)
    start_time = time.time()
    ml_estimator = MLEstimator(
        max_iter=cfg['traditional']['ml']['max_iter'],
        tol=cfg['traditional']['ml']['tol'],
        learning_rate=cfg['traditional']['ml']['learning_rate']
    )
    H_est_ml = ml_estimator.estimate(Y, X)
    results['ml'] = calculate_performance_metrics(H, H_est_ml)
    print_progress(f"ML 估计完成 (耗时: {time.time()-start_time:.2f}s)", 1)
    print_progress("ML 性能指标:", 2)
    for metric, value in results['ml'].items():
        print_progress(f"{metric}: {value:.6f}", 3)
    
    return results

def evaluate_dl_methods(train_loader, val_loader, test_loader, input_size, cfg):
    """
    评估深度学习方法的性能
    
    实现了五种深度学习模型的训练和评估：
    1. CNN（卷积神经网络）
    2. RNN（循环神经网络）
    3. LSTM（长短期记忆网络）
    4. GRU（门控循环单元）
    5. Hybrid（混合CNN-LSTM模型）
    
    对每个模型：
    - 初始化模型结构
    - 配置训练器
    - 执行训练过程
    - 加载最佳模型
    - 在测试集上评估
    - 记录性能指标
    
    参数:
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        test_loader (DataLoader): 测试数据加载器
        input_size (int): 输入特征维度
        cfg (dict): 配置字典，包含模型和训练参数
    
    返回:
        dict: 包含各模型训练结果和性能指标的字典
    """
    device = torch.device('cuda' if cfg['experiment']['use_cuda'] and torch.cuda.is_available() else 'cpu')
    results = {}
    
    # 获取通用模型设置
    common_cfg = cfg['models']['common']
    
    # 定义要评估的模型
    models = {
        'CNN': CNNEstimator(input_size=input_size),
        'RNN': RNNEstimator(
            input_size=input_size,
            hidden_size=cfg['models']['rnn']['hidden_size'],
            num_layers=cfg['models']['rnn']['num_layers']
        ),
        'LSTM': LSTMEstimator(
            input_size=input_size,
            hidden_size=cfg['models']['lstm']['hidden_size'],
            num_layers=cfg['models']['lstm']['num_layers']
        ),
        'GRU': GRUEstimator(
            input_size=input_size,
            hidden_size=cfg['models']['gru']['hidden_size'],
            num_layers=cfg['models']['gru']['num_layers']
        ),
        'Hybrid': HybridEstimator(
            input_size=input_size,
            hidden_size=cfg['models']['hybrid']['rnn_hidden_size']
        )
    }
    
    print_section_header("评估深度学习方法")
    
    # 评估每个模型
    for name, model in models.items():
        print_progress(f"\n开始训练 {name} 模型...", 1)
        print_progress(f"模型结构:", 2)
        print_progress(str(model), 3)
        
        # 创建训练器
        trainer = ChannelEstimatorTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=cfg['experiment']['save_dir'],
            project_name=f"{cfg['experiment']['name']}_{name.lower()}",
            device=device,
            learning_rate=common_cfg['learning_rate']
        )
        
        # 训练模型
        start_time = time.time()
        trainer.train(
            epochs=cfg['training']['epochs'],
            early_stopping_patience=cfg['training']['early_stopping']['patience']
        )
        training_time = time.time() - start_time
        print_progress(f"{name} 模型训练完成 (耗时: {training_time:.2f}s)", 1)
        
        # 加载最佳模型进行测试
        trainer.load_model('best.pt')
        model = trainer.model
        
        # 测试评估
        print_progress(f"开始评估 {name} 模型...", 1)
        model.eval()
        all_preds = []
        all_true = []
        
        test_start_time = time.time()
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                outputs = model(batch_X)
                all_preds.append(outputs.cpu().numpy())
                all_true.append(batch_Y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_true = np.concatenate(all_true, axis=0)
        
        metrics = calculate_performance_metrics(all_true, all_preds)
        results[name] = {
            'metrics': metrics,
            'model': model,
            'trainer': trainer
        }
        
        print_progress(f"{name} 模型评估完成 (耗时: {time.time()-test_start_time:.2f}s)", 1)
        print_progress(f"{name} 模型性能指标:", 2)
        for metric_name, value in metrics.items():
            print_progress(f"{metric_name}: {value:.6f}", 3)
    
    return results

def plot_results(traditional_results, dl_results, cfg):
    """
    绘制实验结果比较图表
    
    生成四种类型的可视化图表：
    1. 整体性能对比图：所有方法的MSE柱状图
    2. 训练历史曲线：每个深度学习模型的训练过程
    3. 性能指标雷达图：传统方法和深度学习方法的多指标对比
    4. 误差分布箱线图：所有方法的MSE分布对比
    
    参数:
        traditional_results (dict): 传统方法的评估结果
        dl_results (dict): 深度学习方法的评估结果
        cfg (dict): 配置字典，包含绘图相关参数
    """
    print_section_header("绘制结果比较图")
    
    # 创建保存目录
    save_dir = Path(cfg['experiment']['plot_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print_progress("生成性能对比图...", 1)
    
    # 1. 整体MSE对比图
    plt.figure(figsize=(12, 6))
    
    # 绘制传统方法结果
    x_traditional = np.arange(len(traditional_results))
    mse_traditional = [results['mse'] for results in traditional_results.values()]
    plt.bar(x_traditional, mse_traditional, alpha=0.8, label='Traditional Methods')
    plt.xticks(x_traditional, traditional_results.keys())
    
    # 绘制深度学习方法结果
    x_dl = np.arange(len(dl_results)) + len(traditional_results)
    mse_dl = [results['metrics']['mse'] for results in dl_results.values()]
    plt.bar(x_dl, mse_dl, alpha=0.8, label='Deep Learning Methods')
    plt.xticks(np.concatenate([x_traditional, x_dl]),
               list(traditional_results.keys()) + list(dl_results.keys()),
               rotation=45)
    
    plt.ylabel('MSE')
    plt.title('所有方法性能对比')
    plt.legend()
    plt.tight_layout()
    
    save_path = save_dir / f'overall_comparison_{timestamp}.png'
    plt.savefig(save_path)
    plt.close()
    
    print_progress(f"整体对比图已保存至: {save_path}", 1)
    
    # 2. 每个深度学习模型的训练历史
    print_progress("生成训练历史图...", 1)
    for name, result in dl_results.items():
        trainer = result['trainer']
        history = trainer.get_history()
        
        plt.figure(figsize=(15, 5))
        
        # 训练损失和验证损失
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{name} 模型训练历史')
        plt.legend()
        plt.grid(True)
        
        # 学习率变化
        plt.subplot(1, 2, 2)
        plt.plot(history['learning_rate'], label='学习率')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('学习率变化')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        save_path = save_dir / f'{name.lower()}_training_history_{timestamp}.png'
        plt.savefig(save_path)
        plt.close()
        
        print_progress(f"{name} 模型训练历史图已保存至: {save_path}", 2)
    
    # 3. 性能指标雷达图
    print_progress("生成性能指标雷达图...", 1)
    metrics = ['mse', 'nmse', 'ber']  # 根据实际指标调整
    
    # 传统方法雷达图
    plt.figure(figsize=(10, 10))
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合图形
    
    ax = plt.subplot(111, polar=True)
    for method, results in traditional_results.items():
        values = [results[metric] for metric in metrics]
        values = np.concatenate((values, [values[0]]))  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2, label=method)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    plt.title('传统方法性能指标对比')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    save_path = save_dir / f'traditional_metrics_radar_{timestamp}.png'
    plt.savefig(save_path)
    plt.close()
    
    print_progress(f"传统方法性能指标雷达图已保存至: {save_path}", 2)
    
    # 深度学习方法雷达图
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    for name, result in dl_results.items():
        values = [result['metrics'][metric] for metric in metrics]
        values = np.concatenate((values, [values[0]]))  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    plt.title('深度学习方法性能指标对比')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    save_path = save_dir / f'dl_metrics_radar_{timestamp}.png'
    plt.savefig(save_path)
    plt.close()
    
    print_progress(f"深度学习方法性能指标雷达图已保存至: {save_path}", 2)
    
    # 4. 箱线图比较
    print_progress("生成性能指标箱线图...", 1)
    plt.figure(figsize=(15, 6))
    
    # 合并所有方法的结果
    all_methods = {}
    all_methods.update({k: {'mse': [v['mse']]} for k, v in traditional_results.items()})
    all_methods.update({k: {'mse': [v['metrics']['mse']]} for k, v in dl_results.items()})
    
    # 创建箱线图数据
    labels = []
    mse_data = []
    for method, data in all_methods.items():
        labels.append(method)
        mse_data.append(data['mse'])
    
    plt.boxplot(mse_data, labels=labels)
    plt.xticks(rotation=45)
    plt.ylabel('MSE')
    plt.title('所有方法MSE分布对比')
    plt.grid(True)
    
    plt.tight_layout()
    save_path = save_dir / f'mse_boxplot_{timestamp}.png'
    plt.savefig(save_path)
    plt.close()
    
    print_progress(f"性能指标箱线图已保存至: {save_path}", 2)
    
    print_progress("所有结果图表已生成完成", 1)

def parse_args():
    """
    解析命令行参数
    
    支持的参数：
    1. --config: 配置文件路径
    2. --device: 运行设备（cpu/cuda）
    3. --seed: 随机种子
    
    返回:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='信道估计方法比较研究')
    parser.add_argument('--config', type=str, default='src/config/default.yml',
                      help='配置文件路径')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                      help='运行设备')
    parser.add_argument('--seed', type=int, help='随机种子')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """
    加载配置文件
    
    功能：
    1. 读取YAML配置文件
    2. 验证必要配置项
    3. 检查配置完整性
    
    参数:
        config_path (str): 配置文件路径
    
    返回:
        dict: 配置字典
    
    异常:
        ValueError: 当缺少必要的配置项时抛出
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 验证必要的配置项
    required_configs = [
        'experiment.name',
        'channel.n_tx',
        'channel.n_rx',
        'channel.n_pilot',
        'channel.snr_db',
        'channel.n_samples',
        'channel.type',
        'traditional.ls.regularization',
        'traditional.ls.lambda',
        'traditional.lmmse.adaptive_snr',
        'traditional.lmmse.correlation_method',
        'traditional.ml.max_iter',
        'traditional.ml.tol'
    ]
    
    for config_path in required_configs:
        value = get_config_value(config, config_path)
        if value is None:
            raise ValueError(f"配置文件缺少必要项: {config_path}")
    
    return config

def get_config_value(config: dict, path: str) -> Any:
    """
    从嵌套字典中获取值
    
    使用点号分隔的路径从嵌套字典中获取值
    例如：'models.cnn.channels' -> config['models']['cnn']['channels']
    
    参数:
        config (dict): 配置字典
        path (str): 以点分隔的配置路径
    
    返回:
        Any: 配置值，如果路径不存在则返回None
    """
    keys = path.split('.')
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='信道估计实验')
    parser.add_argument('--config', type=str, default='src/config/default.yml',
                      help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 创建保存目录
    save_dir = Path(cfg['experiment']['save_dir'])
    plot_dir = Path(cfg['experiment']['plot_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置日志记录
    logger = setup_logging(save_dir, timestamp)
    sys.stdout = logger
    
    # 设置随机种子
    seed = cfg['experiment']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and cfg['experiment']['use_cuda']:
        torch.cuda.manual_seed(seed)
    
    try:
        print_section_header("开始信道估计实验")
        
        # 打印实验参数
        print_progress("实验参数配置:", 1)
        print_progress(f"发射天线数: {cfg['channel']['n_tx']}", 2)
        print_progress(f"接收天线数: {cfg['channel']['n_rx']}", 2)
        print_progress(f"导频长度: {cfg['channel']['n_pilot']}", 2)
        print_progress(f"信噪比: {cfg['channel']['snr_db']} dB", 2)
        print_progress(f"样本数量: {cfg['channel']['n_samples']}", 2)
        print_progress(f"批次大小: {cfg['data']['loader']['batch_size']}", 2)
        print_progress(f"运行设备: {'cuda' if torch.cuda.is_available() and cfg['experiment']['use_cuda'] else 'cpu'}", 2)
        
        # 生成数据
        print_progress("\n初始化数据生成器...", 1)
        start_time = time.time()
        data_generator = ChannelDataGenerator(
            n_tx=cfg['channel']['n_tx'],
            n_rx=cfg['channel']['n_rx'],
            n_pilot=cfg['channel']['n_pilot'],
            channel_type=cfg['channel']['type'],
            rician_k=cfg['channel'].get('rician_k', 1.0)
        )
        
        print_progress("开始生成数据...", 1)
        H, X, Y = data_generator.generate(
            n_samples=cfg['channel']['n_samples'],
            snr_db=cfg['channel']['snr_db']
        )
        
        print_progress(f"信道矩阵 H 形状: {H.shape}", 2)
        print_progress(f"导频符号 X 形状: {X.shape}", 2)
        print_progress(f"接收信号 Y 形状: {Y.shape}", 2)
        print_progress(f"数据生成完成 (耗时: {time.time()-start_time:.2f}s)", 1)
        
        # 数据预处理
        print_progress("\n开始数据预处理...", 1)
        start_time = time.time()
        preprocessor = ChannelPreprocessor(
            normalization=cfg['data']['preprocessing']['normalization'],
            remove_outliers=cfg['data']['preprocessing']['remove_outliers'],
            outlier_threshold=cfg['data']['preprocessing']['outlier_threshold']
        )
        
        H = preprocessor.fit_transform(H)
        Y = preprocessor.transform(Y)
        
        print_progress(f"数据预处理完成 (耗时: {time.time()-start_time:.2f}s)", 1)
        
        # 数据增强
        if cfg['data']['augmentation']['enabled']:
            print_progress("\n开始数据增强...", 1)
            start_time = time.time()
            
            H = augment_data(
                H,
                methods=cfg['data']['augmentation']['methods'],
                noise_std=cfg['data']['augmentation']['noise_std'],
                phase_shift_range=cfg['data']['augmentation']['phase_shift_range'],
                magnitude_scale_range=cfg['data']['augmentation']['magnitude_scale_range']
            )
            
            print_progress("增强后数据形状:", 2)
            print_progress(f"H: {H.shape}", 3)
            print_progress(f"X: {X.shape}", 3)
            print_progress(f"Y: {Y.shape}", 3)
            print_progress(f"数据增强完成 (耗时: {time.time()-start_time:.2f}s)", 1)
        
        # 评估传统方法
        print_section_header("评估传统估计方法")
        traditional_results = evaluate_traditional_methods(
            H=H,
            X=X,
            Y=Y,
            snr=10**(cfg['channel']['snr_db']/10),
            cfg=cfg
        )
        
        # 数据集划分
        print_section_header("准备深度学习数据集")
        n_samples = H.shape[0]
        train_ratio = cfg['data']['split']['train_ratio']
        val_ratio = cfg['data']['split']['val_ratio']
        
        # 计算样本数量
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val
        
        print_progress("数据集划分:", 1)
        print_progress(f"训练集: {n_train} 样本", 2)
        print_progress(f"验证集: {n_val} 样本", 2)
        print_progress(f"测试集: {n_test} 样本", 2)
        
        # 转换为PyTorch张量
        # 将复数数据分离为实部和虚部
        X_real = torch.from_numpy(X.real).float()
        X_imag = torch.from_numpy(X.imag).float()
        H_real = torch.from_numpy(H.real).float()
        H_imag = torch.from_numpy(H.imag).float()
        
        # 在最后一维拼接实部和虚部
        X_tensor = torch.cat([X_real, X_imag], dim=-1)
        H_tensor = torch.cat([H_real, H_imag], dim=-1)
        
        # 创建数据集
        dataset = TensorDataset(X_tensor, H_tensor)
        
        # 划分数据集
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg['data']['loader']['batch_size'],
            shuffle=True,
            num_workers=cfg['data']['loader']['num_workers'],
            pin_memory=cfg['data']['loader']['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg['data']['loader']['batch_size'],
            shuffle=False,
            num_workers=cfg['data']['loader']['num_workers'],
            pin_memory=cfg['data']['loader']['pin_memory']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg['data']['loader']['batch_size'],
            shuffle=False,
            num_workers=cfg['data']['loader']['num_workers'],
            pin_memory=cfg['data']['loader']['pin_memory']
        )
        
        # 计算输入大小（考虑实部和虚部）
        input_size = X.shape[1] * X.shape[2] * 2  # n_tx * n_pilot * 2 (实部和虚部)
        
        # 训练和评估深度学习模型
        dl_results = evaluate_dl_methods(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            input_size=input_size,
            cfg=cfg
        )
        
        # 保存结果
        results = {
            'config': cfg,
            'traditional_results': traditional_results,
            'dl_results': {
                name: {
                    'metrics': result['metrics']
                } for name, result in dl_results.items()
            }
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = save_dir / f"results_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        # 绘制结果
        plot_results(traditional_results, dl_results, cfg)
        
        print("\n实验完成！结果已保存至:", result_file)
    
    except Exception as e:
        print(f"\n实验出错！错误信息：{str(e)}")
        raise e
    
    finally:
        # 恢复标准输出
        sys.stdout = sys.__stdout__
        # 关闭日志文件
        logger.log_file.close()

if __name__ == "__main__":
    main() 