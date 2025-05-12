# 跨平台可视化指南

PhaseFieldX 现在提供了跨平台的可视化功能，可以在不同的操作系统上工作，包括 macOS、Windows 和 Linux。这个新的可视化模块支持多种后端，使您可以根据自己的需求和平台选择最合适的可视化工具。

## 可视化后端

新的可视化模块支持以下后端：

1. **Plotly**：基于 Web 的交互式 3D 可视化，可在任何平台上工作，无需 OpenGL 依赖。
2. **PyVista**：基于 VTK 的 3D 可视化，提供高质量的渲染，但在某些平台上可能需要 OpenGL 支持。
3. **Matplotlib**：基础的 2D 可视化，在所有平台上都能工作，但 3D 功能有限。

## 安装依赖

要使用跨平台可视化功能，您需要安装以下依赖：

```bash
# 基本依赖
conda install -c conda-forge matplotlib

# 推荐安装 Plotly 以获得最佳的跨平台体验
pip install plotly

# 如果您的平台支持 OpenGL，也可以安装 PyVista
conda install -c conda-forge pyvista
```

## 使用方法

### 基本用法

最简单的方法是使用 `plot_vtu` 函数，它会自动选择最合适的后端：

```python
from phasefieldx.PostProcessing.visualization import plot_vtu

# 可视化 VTU 文件
plot_vtu("path/to/your/file.vtu", scalars="u", title="Displacement Field")
```

### 指定后端

您可以明确指定要使用的后端：

```python
from phasefieldx.PostProcessing.visualization import Visualizer

# 使用 Plotly 后端
visualizer = Visualizer(backend='plotly')
visualizer.plot_vtu("path/to/your/file.vtu", scalars="u", title="Plotly Backend")

# 使用 PyVista 后端
visualizer = Visualizer(backend='pyvista')
visualizer.plot_vtu("path/to/your/file.vtu", scalars="u", title="PyVista Backend")

# 使用 Matplotlib 后端
visualizer = Visualizer(backend='matplotlib')
visualizer.plot_vtu("path/to/your/file.vtu", scalars="u", title="Matplotlib Backend")
```

### 在现有代码中替换 PyVista

如果您有使用 PyVista 的现有代码，可以轻松地替换为新的可视化模块：

原始代码：
```python
import pyvista as pv

# 读取 VTU 文件
pv.start_xvfb()
file_vtu = pv.read("path/to/your/file.vtu")
file_vtu.plot(scalars='u', cpos='xy', show_scalar_bar=True, show_edges=False)
```

替换为：
```python
from phasefieldx.PostProcessing.visualization import plot_vtu

# 使用跨平台可视化模块
plot_vtu("path/to/your/file.vtu", scalars='u', cpos='xy', show_scalar_bar=True, show_edges=False)
```

## 高级功能

### 保存可视化结果

您可以将可视化结果保存为图像文件：

```python
from phasefieldx.PostProcessing.visualization import plot_vtu

# 保存为图像文件
plot_vtu("path/to/your/file.vtu", scalars="u", save_path="output.png")
```

### 自定义可视化

您可以通过传递额外的参数来自定义可视化效果：

```python
from phasefieldx.PostProcessing.visualization import plot_vtu

# 自定义可视化
plot_vtu(
    "path/to/your/file.vtu",
    scalars="u",
    cpos="xy",  # 相机位置：'xy', 'xz', 或 'yz'
    show_scalar_bar=True,  # 显示颜色条
    show_edges=True,  # 显示网格边缘
    title="自定义可视化"  # 设置标题
)
```

## 示例

查看 `examples/visualization_example.py` 文件，了解如何使用不同的后端进行可视化。

## 故障排除

如果您在使用可视化模块时遇到问题，请尝试以下解决方案：

1. **安装依赖**：确保您已安装所需的依赖（Plotly 和/或 PyVista）。
2. **更新依赖**：尝试更新到最新版本的依赖。
3. **尝试不同的后端**：如果一个后端不工作，尝试使用另一个后端。
4. **检查 VTU 文件**：确保 VTU 文件存在且格式正确。

如果问题仍然存在，请在 GitHub 上提交 issue。
