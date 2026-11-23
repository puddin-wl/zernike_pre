"""显式选择图形后端，避免 IDE 或此前的无窗口运行改变显示行为。"""

import os
from pathlib import Path
import sys


_dll_handles = {}


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
    else:
        # 本项目环境已安装 PySide6。只影响本次 Python 进程。
        os.environ["QT_API"] = "pyside6"
        matplotlib.use("QtAgg", force=True)
