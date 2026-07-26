"""
绘图与终端输出的中文支持工具

- setup_chinese_font(): 自动探测系统中文字体并配置 matplotlib，
  解决中文绘图乱码问题（Windows 优先微软雅黑，macOS 优先苹方，
  Linux 优先 Noto/文泉驿，均无则回退 DejaVu 并给出提示）
- disp_width() / pad_cjk(): 计算中英文混排字符串的显示宽度并按宽度填充，
  用于终端表格对齐（CJK 字符占两个显示宽度）
"""

import unicodedata

_font_ready = False


def setup_chinese_font() -> bool:
    """
    配置 matplotlib 中文字体（幂等，可重复调用）

    返回:
        bool: 是否成功找到中文字体
    """
    global _font_ready
    if _font_ready:
        return True

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        'Microsoft YaHei',      # Windows 微软雅黑
        'SimHei',               # Windows 黑体
        'PingFang SC',          # macOS 苹方
        'Hiragino Sans GB',     # macOS 冬青黑体
        'Noto Sans CJK SC',     # Linux Noto
        'WenQuanYi Micro Hei',  # Linux 文泉驿
        'Arial Unicode MS',
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [f for f in candidates if f in available]

    if chosen:
        plt.rcParams['font.sans-serif'] = chosen + ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
        _font_ready = True
    else:
        print("提示: 未找到中文字体，图表中的中文可能显示为方块；"
              "建议安装微软雅黑/Noto Sans CJK 等中文字体。")
    return _font_ready


def disp_width(text: str) -> int:
    """计算字符串的终端显示宽度（CJK 字符计 2，其余计 1）"""
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1
               for c in str(text))


def pad_cjk(text: str, width: int, align: str = '<') -> str:
    """
    按显示宽度填充字符串（用于中英文混排的终端表格对齐）

    参数:
        text: 原始字符串
        width: 目标显示宽度
        align: '<' 左对齐，'>' 右对齐，'^' 居中
    """
    text = str(text)
    padding = max(0, width - disp_width(text))
    if align == '>':
        return ' ' * padding + text
    if align == '^':
        left = padding // 2
        return ' ' * left + text + ' ' * (padding - left)
    return text + ' ' * padding
