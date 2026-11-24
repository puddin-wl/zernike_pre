"""验证新入口、无绘图理论程序以及已删除选项不会被继续接受。"""

import json
from pathlib import Path
import subprocess
import sys

import pytest
from optics.cli import _parser


def test_theory_entry_point_does_not_import_plotting_or_fft(tmp_path):
    code = (
        "import runpy,sys; "
        f"sys.argv=['resolution.py','--lens-diameter-mm','6','--immersion-medium','oil',"
        f"'--sample-refractive-index','1.40','--output',{str(tmp_path)!r}]; "
        "runpy.run_path('resolution.py',run_name='__main__'); "
        "assert not any(n in sys.modules for n in ('matplotlib','numpy','scipy'))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    summary = json.loads((tmp_path/"summary.json").read_text(encoding="utf-8"))
    assert summary["parameters"]["beam_diameter_mm"] == 6
    assert summary["parameters"]["immersion_refractive_index"] == 1.515
    assert summary["results"]["sample_medium"]["refractive_index"] == 1.40
    assert "1.37" not in result.stdout
    assert set(p.name for p in tmp_path.iterdir()) == {"summary.json", "summary.txt"}


@pytest.mark.parametrize("program,options", [
    ("fourier_focus", ["--beam", "uniform"]),
    ("zernike_focus", ["--zernike-radius-mm", "2"]),
    ("fourier_focus", ["--window-mm", "8"]),
    ("fourier_focus", ["--zernike", "2,0,0.1"]),
])
def test_removed_options_are_rejected(program, options):
    with pytest.raises(SystemExit):
        _parser(program).parse_args(options)


def test_axial_scan_options_are_available_in_both_focus_programs():
    for program in ("fourier_focus", "zernike_focus"):
        args = _parser(program).parse_args(["--axial-samples", "97", "--axial-half-range-um", "1500"])
        assert args.axial_samples == 97
        assert args.axial_half_range_um == 1500


def test_numeric_initial_index_alias_and_sample_option_scope():
    assert _parser("resolution").parse_args(["--immersion-refractive-index", "1.42"]).immersion_medium == 1.42
    for program in ("fourier_focus", "zernike_focus"):
        for options in (["--sample-refractive-index", "1.40"], ["--immersion-medium", "oil"]):
            with pytest.raises(SystemExit):
                _parser(program).parse_args(options)
