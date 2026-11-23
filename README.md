# Zernike Pre：衍射分辨率与像差模拟

两个独立程序，共用空气中的标量近轴波动光学核心。观察面固定在薄透镜后 **z=f**，不搜索焦面、不重新调焦。主程序只保留输入参数。

轴向指标采用光强 FWHM（50% 峰值的全宽），并非未指定判据的通用 depth of focus。物理假设、系数归一化及数值精度说明见 [docs/model.md](docs/model.md)。

## 安装

安装 Conda 后，在终端执行：

```powershell
git clone https://github.com/puddin-wl/zernike_pre.git
cd zernike_pre
conda env create -f environment.yml
conda activate lpr
```

依赖包括 Python 3.12、NumPy、SciPy、Matplotlib、PySide6 和 pytest。Windows 和 macOS 共用计算程序，默认显示图窗，也可用 `--no-show` 只保存图片与数据。

### macOS（Apple Silicon / Intel）

在 Mac 的终端创建本机环境后运行：

```bash
conda env create -f environment.yml
conda activate lpr
python diffraction.py
python aberration.py
```

已有环境可运行 `conda env update -f environment.yml`。**不要在 Mac 使用 `environment.win-64.lock.txt`**，该文件仅包含 Windows 构建版本；Mac 应使用通用的 `environment.yml`，由 Conda 解析本机架构的依赖。

显示优先使用跨平台 QtAgg；macOS 在 Qt 不可用时尝试原生 MacOSX 后端，再尝试 TkAgg。无法启用窗口时会明确报错，不会自动悄悄切成仅保存模式。Mac 自动选择已安装的苹方、黑体或宋体等中文字体；Windows 保留微软雅黑选择和 Conda Qt DLL 路径修复。

不使用 Conda 时，也可在 Python 3.12 虚拟环境中执行 `python -m pip install -r requirements.txt`。Qt 支持范围以本机可安装的 PySide6 版本为准，参见 [Qt for Python 文档](https://doc.qt.io/qtforpython-6/gettingstarted.html)。

## 程序一：无像差

修改 `diffraction.py` 中的参数，然后运行：

```powershell
conda run --no-capture-output -n lpr python diffraction.py
```

输入波长、透镜有效直径、焦距、光束类型和光束直径。支持均匀圆光束和高斯光束，不提供像差输入。输出横向 x/y 与轴向光强 FWHM，绘制输入光强、焦斑、XZ 光强切面、横向截面和轴向曲线。无像差圆对称时 x=y，只画一个纵向切面和一条横向曲线。

## 程序二：加载像差

修改 `aberration.py` 中的参数，然后运行：

```powershell
conda run --no-capture-output -n lpr python aberration.py
```

除相同的光学参数外，输入归一化圆半径和 Zernike 系数。系数使用 **waves RMS**，例如：

```python
ZernikeTerm(n=3, m=1, coefficient_waves=0.15)
```

`1 wave = 2π rad RMS`。相位图采用 rad 显示。程序只计算加载后自己的光场，不同时计算无像差参考。输出输入光强、输入相位、z=f 焦斑、XZ/YZ 光强切面、横向 x/y 截面和轴向曲线。

切面穿过 z=f 光斑主峰：XZ 固定 y，YZ 固定 x，各自在标题标出固定坐标；不随 z 追踪峰值。两张切面共用同一线性色标，不逐行重新归一化，虚线表示 z=f。轴向扫描范围仍由 `--axial-half-range-um` 控制。切面数组和坐标一起存入 `fields.npz`。

将所有系数置零或使用 `--no-aberrations`，在其他参数及数值设置相同时，应与程序一得到相同结果。

## 命令行输入

两份主程序的参数也可以由命令行覆盖：

```powershell
# 两种入射光都算一遍，只保存图。
conda run --no-capture-output -n lpr python diffraction.py --beam both --no-show
conda run --no-capture-output -n lpr python aberration.py --beam both --no-show

# 高斯光直径大于透镜口径。
conda run --no-capture-output -n lpr python diffraction.py --wavelength-nm 532 --lens-diameter-mm 4 --focal-length-mm 50 --beam-diameter-mm 8 --no-show

# 加载彗差和离焦；命令行列表替代主程序中的系数列表。
conda run --no-capture-output -n lpr python aberration.py --zernike 3,1,0.15 --zernike 2,0,0.1 --no-show

# 全部系数清零。
conda run --no-capture-output -n lpr python aberration.py --no-aberrations --no-show
```

直接运行两份脚本时，默认使用交互图窗显示，程序等待关闭图窗后结束；同时自动保存图片。只有显式传入 `--no-show` 才不弹窗。命令行会打印 `overview.png` 的完整保存路径。`--help` 查看全部选项；`--verbose` 显示进度和数值诊断。

VS Code 打开项目文件夹后，执行“Python: Select Interpreter”并选择 `lpr`，再使用右上角“运行 Python 文件”。程序会自动补齐该 Conda 环境的 Windows Qt DLL 搜索路径，不要求终端事先执行 `conda activate`。F5 也提供了对应的项目启动配置，使用当前选择的解释器。本机 `.vscode/settings.json` 不纳入仓库。

## 输出与目录

默认按程序和光束类型保存，每次更新同目录下的结果，不再不断新增运行文件夹：

```text
zernike_pre/
├─ diffraction.py              程序一：无像差
├─ aberration.py               程序二：加载像差
├─ optics/                     共用计算与绘图封装
│  ├─ config.py                输入参数与单位
│  ├─ propagation.py           Fresnel 传播与薄透镜
│  ├─ zernike.py               RMS 归一化基底
│  ├─ metrics.py               FWHM 等指标
│  ├─ simulation.py            两个独立计算接口及对比接口
│  ├─ cli.py / report.py       独立程序输入与输出
│  └─ comparison*.py           原对比程序，单独封装保留
├─ tests/                      物理和程序一致性检查
├─ docs/model.md               物理假设与精度说明
├─ results/                    运行时生成，不纳入仓库
│  ├─ diffraction/gaussian/    程序一的高斯结果
│  └─ aberration/gaussian/     程序二的高斯结果
└─ archive/                    仅本机历史归档，不纳入仓库
```

每份结果包括 `overview.png`、`overview.svg`、`summary.json` 和 `fields.npz`。`--output 路径` 可指定输出目录。强度图按本次光场的自身峰值归一化，实际功率保存在 JSON 中。

原有对比功能通过以下命令运行，参数在 `optics/comparison.py` 中：

```powershell
conda run --no-capture-output -n lpr python -m optics.comparison
```

## 环境和验证

依赖由 `environment.yml` 描述，精确 Windows 包版本保存在 `environment.win-64.lock.txt`。需要复现相同 Windows 构建版本时，可在环境不存在的情况下执行：

```powershell
# 按锁定文件创建 Windows 环境。
conda create -n lpr --file environment.win-64.lock.txt
conda env config vars set -n lpr PYTHONNOUSERSITE=1

# 运行验证。
conda run --no-capture-output -n lpr python -m pytest -q
```

检查覆盖解析高斯传播、截光功率、Airy 尺度、Zernike 正交归一、XZ/YZ 固定切面、零像差时两程序一致性、显示后端和平台字体选择。GitHub Actions 分别在 Windows、Apple Silicon Mac 和 Intel Mac 上运行检查及两份程序的图片导出。
