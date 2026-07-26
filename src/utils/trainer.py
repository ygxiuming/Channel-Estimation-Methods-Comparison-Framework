"""
信道估计训练器模块（YOLO 风格训练输出）

仿 Ultralytics YOLO 的训练体验：
- 训练开始时打印配置摘要与列表头（Epoch / train_loss / val_loss / val_NMSE / lr）
- 每个 epoch：动态批次进度条（结束后清除）+ 一行汇总指标
- 验证集 NMSE(dB) 作为核心精度指标（类比 YOLO 的 mAP）
- 权重保存: weights/best.pt（验证损失最优）与 weights/last.pt（最后一轮）
- 训练日志: results.csv（逐 epoch 记录 loss/NMSE/lr/耗时）
- 训练结束自动生成 results.png（损失曲线 + 学习率曲线）
- 早停机制与 YOLO 风格提示
"""

import csv
import time
from pathlib import Path

import torch
from tqdm import tqdm

from utils.plotting import setup_chinese_font


class ChannelEstimatorTrainer:
    """
    信道估计模型训练器

    参数:
        model (nn.Module): 要训练的模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        save_dir (str): 结果保存根目录
        project_name (str): 实验名称（创建同名子目录）
        device (str): 训练设备（'cuda' 或 'cpu'）
        learning_rate (float): 初始学习率
        use_tensorboard (bool): 是否启用 TensorBoard（不可用时自动降级）
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        save_dir='runs/train',
        project_name='channel_estimation',
        device='cuda',
        learning_rate=0.001,
        use_tensorboard=True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate

        # 保存目录结构（YOLO 风格）: save_dir/project_name/weights/
        self.save_dir = Path(save_dir) / project_name
        self.weights_dir = self.save_dir / 'weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard（非ASCII路径等环境下可能不可用，优雅降级）
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.save_dir / 'tensorboard'))
            except Exception as e:
                print(f"Warning: TensorBoard unavailable ({e}), disabled.")

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5
        )

        # 训练状态
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.current_epoch = 0
        self.total_epochs = 1  # train() 开始时会被更新
        self.history = {'train_loss': [], 'val_loss': [],
                        'val_nmse_db': [], 'learning_rate': []}

        # YOLO 风格 results.csv 日志
        self.results_csv = self.save_dir / 'results.csv'
        with open(self.results_csv, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(
                ['epoch', 'time', 'train_loss', 'val_loss', 'val_nmse_db', 'lr'])

    def train_epoch(self):
        """训练一个 epoch（动态进度条，结束后清除，保持表格整洁）"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader,
                    desc=f'    {self.current_epoch + 1}/{self.total_epochs}',
                    leave=False, ncols=110,
                    bar_format='{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]')

        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X), y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{total_loss / (batch_idx + 1):.5f}'})

            if self.writer is not None:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), step)

        pbar.close()
        return total_loss / len(self.train_loader)

    def validate(self):
        """
        验证模型（静默，无进度条）

        返回:
            (val_loss, val_nmse_db): 验证损失与验证集 NMSE(dB)
            NMSE = val_loss / mean(target²)，与复数域 NMSE 定义一致
        """
        self.model.eval()
        total_loss = 0.0
        total_power = 0.0

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                loss = self.criterion(self.model(X), y)
                total_loss += loss.item()
                total_power += y.pow(2).mean().item()

        val_loss = total_loss / len(self.val_loader)
        mean_power = total_power / len(self.val_loader)
        nmse = val_loss / mean_power if mean_power > 0 else float('inf')
        nmse_db = 10 * torch.log10(torch.tensor(nmse)).item() if nmse > 0 else -float('inf')
        return val_loss, nmse_db

    def train(self, epochs=100, early_stopping_patience=10):
        """完整训练流程（YOLO 风格输出）"""
        self.total_epochs = epochs
        early_stopping_counter = 0
        start_time = time.time()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n模型概况 (Model summary): 参数量 {n_params:,}, "
              f"设备={self.device}, 批大小={self.train_loader.batch_size}, "
              f"学习率={self.learning_rate}, 轮数={epochs}\n")

        print(f"{'轮次(Epoch)':>10} {'训练损失(train_loss)':>18} "
              f"{'验证损失(val_loss)':>18} {'验证NMSE(dB)':>14} {'学习率(lr)':>12}")

        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                epoch_start = time.time()

                train_loss = self.train_epoch()
                val_loss, val_nmse_db = self.validate()

                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)

                # 记录历史与 CSV 日志
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_nmse_db'].append(val_nmse_db)
                self.history['learning_rate'].append(current_lr)
                with open(self.results_csv, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([
                        epoch + 1, f'{time.time() - start_time:.2f}',
                        f'{train_loss:.6f}', f'{val_loss:.6f}',
                        f'{val_nmse_db:.3f}', f'{current_lr:.3e}'])

                if self.writer is not None:
                    self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
                    self.writer.add_scalar('val/epoch_loss', val_loss, epoch)
                    self.writer.add_scalar('val/nmse_db', val_nmse_db, epoch)
                    self.writer.add_scalar('train/lr', current_lr, epoch)

                # 一行汇总（YOLO 风格）
                print(f"{epoch + 1:>6}/{epochs:<4} {train_loss:>12.5f} "
                      f"{val_loss:>12.5f} {val_nmse_db:>8.2f}dB {current_lr:>11.2e}"
                      f"  [{time.time() - epoch_start:.1f}s]")

                # 保存 best / last 权重（YOLO 风格）
                self.save_model('last.pt')
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    self.save_model('best.pt')
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_patience:
                    print(f"\n早停 (EarlyStopping): 连续 {early_stopping_patience} 轮 "
                          f"验证损失无改善，训练提前结束。"
                          f"最优结果出现在第 {self.best_epoch} 轮，"
                          f"最优验证损失 {self.best_val_loss:.6f}。")
                    break

        except KeyboardInterrupt:
            print('\n训练被用户中断 (Training interrupted by user).')

        finally:
            self.save_model('last.pt')
            if self.writer is not None:
                self.writer.close()
            self._plot_results()

            elapsed = time.time() - start_time
            print(f"\n训练完成 (Training completed)，耗时 {elapsed / 3600:.3f} 小时。")
            print(f"最优验证损失 {self.best_val_loss:.6f}，"
                  f"出现在第 {self.best_epoch} 轮。")
            print(f"结果已保存至 (Results saved to): {self.save_dir}\n")

    def _plot_results(self):
        """训练结束后生成 results.png（损失曲线 + 学习率曲线，YOLO 风格）"""
        if not self.history['train_loss']:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            epochs = range(1, len(self.history['train_loss']) + 1)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            axes[0].plot(epochs, self.history['train_loss'], label='train')
            axes[0].plot(epochs, self.history['val_loss'], label='val')
            axes[0].set_yscale('log')
            axes[0].set_xlabel('epoch')
            axes[0].set_title('loss')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            axes[1].plot(epochs, self.history['val_nmse_db'], color='green')
            axes[1].set_xlabel('epoch')
            axes[1].set_title('val NMSE (dB)')
            axes[1].grid(alpha=0.3)

            axes[2].plot(epochs, self.history['learning_rate'], color='orange')
            axes[2].set_xlabel('epoch')
            axes[2].set_title('learning rate')
            axes[2].grid(alpha=0.3)

            fig.tight_layout()
            fig.savefig(self.save_dir / 'results.png', dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: failed to plot results.png ({e})")

    def save_model(self, filename):
        """保存完整训练状态（模型/优化器/调度器/历史）"""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, str(self.weights_dir / filename))

    def load_model(self, filename):
        """加载训练状态，成功返回 True"""
        load_path = self.weights_dir / filename
        if load_path.exists():
            checkpoint = torch.load(str(load_path), weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            if 'history' in checkpoint:
                self.history = checkpoint['history']
            return True
        return False

    def get_history(self):
        """获取训练历史"""
        return self.history
