"""显式选择图形后端，避免 IDE 或此前的无窗口运行改变显示行为。"""

import os
import importlib.util
from pathlib import Path
import sys


_dll_handles = {}


def _interactive_backends(platform):
    # Qt 在 Windows/macOS 都可用；Mac 另有原生 Cocoa 后端。
    return ("QtAgg", "MacOSX", "TkAgg") if platform == "darwin" else ("QtAgg", "TkAgg")


def _select_interactive_backend(switch_backend, platform):
    failures = []
    for backend in _interactive_backends(platform):
        try:
            switch_backend(backend)
            return backend
        except (ImportError, RuntimeError) as error:
            failures.append(f"{backend}: {error}")
    raise RuntimeError(
        "无法启用图形窗口。请确认已安装 environment.yml 中的依赖，并在桌面终端或 VS Code 中运行。"
        "仅保存图片可添加 --no-show。\n"+"\n".join(failures)
    )


def _font_candidates(platform):
    if platform == "darwin":
        return ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS",
                "Noto Sans CJK SC", "DejaVu Sans")
    if platform == "win32":
        return ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans")
    return ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "Source Han Sans SC", "DejaVu Sans")


def font_families(platform=None):
    """仅选择本机实际安装的字体，避免 Mac 上反复查找微软雅黑。"""
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = [name for name in _font_candidates(platform or sys.platform) if name in available]
    return selected or ["DejaVu Sans"]


def _configure_windows_runtime():
    """VS Code 可直接调用 Conda 的 python.exe，而不执行 conda activate。

    此时必须显式提供当前环境的 Qt DLL 路径，避免加载 PATH 中其他
    Anaconda/Qt 安装的 DLL。句柄保留到进程结束，不能随函数返回释放。
    """
    if os.name != "nt":
        return
    library_bin = Path(sys.prefix)/"Library"/"bin"
    if not (library_bin/"Qt6Core.dll").is_file():
        return
    directory = str(library_bin)
    if directory not in _dll_handles:
        _dll_handles[directory] = os.add_dll_directory(directory)
    paths = os.environ.get("PATH", "").split(os.pathsep)
    if not paths or os.path.normcase(os.path.normpath(paths[0])) != os.path.normcase(directory):
        os.environ["PATH"] = directory+os.pathsep+os.environ.get("PATH", "")


def configure_display(no_show=False):
    # Matplotlib 可能在读取外部 MPLBACKEND 时提前探测 Qt；仅补 DLL 路径，
    # 不创建窗口，Agg 模式也可以安全使用。
    _configure_windows_runtime()
    import matplotlib

    if no_show:
        matplotlib.use("Agg", force=True)
        return "Agg"
    if importlib.util.find_spec("PySide6") is not None:
        os.environ["QT_API"] = "pyside6"
    # 真正导入后端以检查可用性，而不是等计算结束才在创建窗口时发现缺依赖。
    import matplotlib.pyplot as plt
    return _select_interactive_backend(plt.switch_backend, sys.platform)


def prepare_interactive_figure(figure):
    """导出之后设置 Qt 图窗：缩小外框，画布保留高度，通过滚动查看。

    Qt 的屏幕和控件尺寸均使用逻辑像素；不额外乘 Retina 缩放倍率。
    这里只调整交互窗口，不改变此前保存的 overview 文件。
    """
    import matplotlib

    if matplotlib.get_backend().lower() != "qtagg":
        return False
    from matplotlib.backends.qt_compat import QtCore, QtWidgets

    canvas = figure.canvas
    window = canvas.manager.window
    if isinstance(window.centralWidget(), QtWidgets.QScrollArea):
        return True
    screen = window.screen() or QtWidgets.QApplication.primaryScreen()
    available = screen.availableGeometry()
    window_width = int(available.width()*0.92)
    window_height = int(available.height()*0.90)

    # 仅让画布宽度适应屏幕。高度至少保留导出图所需的空间，避免三行
    # 子图被窗口管理器压扁；极窄的屏幕同时允许横向滚动。
    original_dpi = figure._original_dpi
    width = max(1000, min(round(figure.get_figwidth()*original_dpi), window_width-32))
    height = max(1500, round(figure.get_figheight()*original_dpi))
    class FigureScrollArea(QtWidgets.QScrollArea):
        def eventFilter(self, watched, event):
            if watched is canvas and event.type() == QtCore.QEvent.Type.Wheel:
                # Matplotlib 会接收画布上的滚轮事件；转交给滚动区，
                # 使鼠标位于图内时，滚轮和触控板也能浏览整张图。
                self.wheelEvent(event)
                return True
            return super().eventFilter(watched, event)

    scroll = FigureScrollArea(window)
    scroll.setWidgetResizable(False)
    scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop)
    window.takeCentralWidget()
    canvas.setFixedSize(width, height)
    scroll.setWidget(canvas)
    canvas.installEventFilter(scroll)
    window.setCentralWidget(scroll)
    window.resize(window_width, window_height)
    window.move(available.x()+(available.width()-window_width)//2,
                available.y()+(available.height()-window_height)//2)
    canvas.draw_idle()
    return True
