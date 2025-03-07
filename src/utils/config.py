"""
配置加载器模块
用于加载和管理YAML配置文件
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

class Config:
    """配置加载器类"""
    
    def __init__(self, config_path: str = "config/default.yml"):
        """
        初始化配置加载器
        
        参数:
            config_path: 配置文件路径，默认为'config/default.yml'
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        参数:
            key: 配置键，使用点号分隔层级，如'models.cnn.channels'
            default: 默认值，当键不存在时返回
        
        返回:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        参数:
            key: 配置键，使用点号分隔层级
            value: 要设置的值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, save_path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        参数:
            save_path: 保存路径，默认为原配置文件路径
        """
        save_path = Path(save_path) if save_path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=False)
    
    def update_from_args(self, args: Dict[str, Any]) -> None:
        """
        从命令行参数更新配置
        
        参数:
            args: 命令行参数字典
        """
        for key, value in args.items():
            if value is not None:  # 只更新非None的值
                self.set(key, value)
    
    def __getitem__(self, key: str) -> Any:
        """使配置对象可以使用字典方式访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """使配置对象可以使用字典方式设置值"""
        self.set(key, value)
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False)
    
    @property
    def experiment(self) -> Dict:
        """获取实验配置"""
        return self.config.get('experiment', {})
    
    @property
    def channel(self) -> Dict:
        """获取信道配置"""
        return self.config.get('channel', {})
    
    @property
    def data(self) -> Dict:
        """获取数据处理配置"""
        return self.config.get('data', {})
    
    @property
    def models(self) -> Dict:
        """获取模型配置"""
        return self.config.get('models', {})
    
    @property
    def training(self) -> Dict:
        """获取训练配置"""
        return self.config.get('training', {})
    
    @property
    def evaluation(self) -> Dict:
        """获取评估配置"""
        return self.config.get('evaluation', {})
    
    @property
    def tensorboard(self) -> Dict:
        """获取TensorBoard配置"""
        return self.config.get('tensorboard', {}) 