"""在独立进程中验证外部后端设置不会改变程序的显示约定。"""

import os
from pathlib import Path
import subprocess
import sys

import pytest


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
