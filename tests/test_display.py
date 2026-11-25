"""在独立进程中验证外部后端设置不会改变程序的显示约定。"""

import os
from pathlib import Path
import subprocess
import sys

import pytest
from optics import display


@pytest.mark.parametrize("no_show,environment_backend,expected", [
    (False, "Agg", "qtagg"),
    (True, "QtAgg", "agg"),
])
def test_explicit_display_mode_overrides_inherited_backend(no_show, environment_backend, expected):
    environment = dict(os.environ, MPLBACKEND=environment_backend, QT_QPA_PLATFORM="offscreen")
    # 模拟 VS Code 右上角直接运行：使用环境内解释器，但不激活 Conda。
    runtime_bin = os.path.normcase(os.path.normpath(str(Path(sys.prefix)/"Library"/"bin")))
    environment["PATH"] = os.pathsep.join(
        entry for entry in environment.get("PATH", "").split(os.pathsep)
        if os.path.normcase(os.path.normpath(entry)) != runtime_bin
    )
    code = (
        "from optics.display import configure_display; "
        f"configure_display(no_show={no_show!r}); "
        "import matplotlib; import matplotlib.pyplot as plt; "
        f"assert matplotlib.get_backend().lower() == {expected!r}; "
        "figure=plt.figure(); figure.canvas.draw(); plt.close(figure)"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout+completed.stderr


def test_macos_falls_back_to_native_backend_if_qt_is_missing():
    attempted = []

    def switch_backend(name):
        attempted.append(name)
        if name == "QtAgg":
            raise ImportError("Qt unavailable")

    assert display._select_interactive_backend(switch_backend, "darwin") == "MacOSX"
    assert attempted == ["QtAgg", "MacOSX"]


def test_display_failure_does_not_silently_select_file_only_backend():
    def unavailable(name):
        raise ImportError(f"{name} unavailable")

    with pytest.raises(RuntimeError, match="--no-show"):
        display._select_interactive_backend(unavailable, "darwin")


@pytest.mark.parametrize("platform,available,expected", [
    ("darwin", ["PingFang SC", "DejaVu Sans"], ["PingFang SC", "DejaVu Sans"]),
    ("darwin", ["Songti SC", "DejaVu Sans"], ["Songti SC", "DejaVu Sans"]),
    ("win32", ["Microsoft YaHei", "DejaVu Sans"], ["Microsoft YaHei", "DejaVu Sans"]),
    ("darwin", ["DejaVu Sans"], ["DejaVu Sans"]),
])
def test_fonts_are_selected_from_the_target_system(monkeypatch, platform, available, expected):
    from types import SimpleNamespace
    from matplotlib import font_manager
    monkeypatch.setattr(font_manager.fontManager, "ttflist", [SimpleNamespace(name=name) for name in available])
    assert display.font_families(platform) == expected


@pytest.mark.parametrize("scale_factor", ["1", "2"])
def test_small_screen_scrolls_without_compressing_plots(scale_factor):
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR=scale_factor)
    code = """
from types import SimpleNamespace
from optics.display import configure_display, prepare_interactive_figure
configure_display()
import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
figure, axes = plt.subplots(3, 2, figsize=(14, 15), layout='constrained')
for axis in axes.flat:
    axis.plot([0, 1], [0, 1])
window = figure.canvas.manager.window
# 模拟笔记本屏幕的逻辑尺寸；2 倍缩放时不应再将尺寸乘二。
window.screen = lambda: SimpleNamespace(availableGeometry=lambda: QtCore.QRect(0, 0, 1280, 800))
assert prepare_interactive_figure(figure)
window.show()
application = QtWidgets.QApplication.instance()
application.processEvents()
figure.canvas.draw()
scroll = window.centralWidget()
assert isinstance(scroll, QtWidgets.QScrollArea)
assert window.width() <= 1280 and window.height() <= 800
assert figure.canvas.width() >= 1000 and figure.canvas.height() >= 1500
assert scroll.verticalScrollBar().maximum() > 0
# 鼠标在画布上时，滚轮/触控板应滚动视口，而不被 Matplotlib 吞掉。
wheel = QtGui.QWheelEvent(QtCore.QPointF(20, 20), QtCore.QPointF(20, 20),
                         QtCore.QPoint(0, -120), QtCore.QPoint(0, -120),
                         QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
                         QtCore.Qt.ScrollPhase.ScrollUpdate, False)
application.sendEvent(figure.canvas, wheel)
application.processEvents()
assert scroll.verticalScrollBar().value() > 0
original_size = figure.get_size_inches().copy()
original_height = axes[0, 0].get_window_extent().height / figure.canvas.device_pixel_ratio
assert original_height > 250
window.resize(900, 500)
application.processEvents()
figure.canvas.draw()
assert max(abs(figure.get_size_inches()-original_size)) < 0.01
assert abs(axes[0, 0].get_window_extent().height / figure.canvas.device_pixel_ratio-original_height) < 1
scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
application.processEvents()
assert scroll.verticalScrollBar().value() > 0
assert prepare_interactive_figure(figure)  # 重复调用不嵌套滚动容器。
plt.close(figure)
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout+completed.stderr
