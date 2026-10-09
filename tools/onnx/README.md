# 面向 WebGPU 的 ONNX 导出

在 PixRestore 仓库根目录、已有的 CUDA PyTorch 环境中运行命令。本地可使用 `conda activate diffusion`。
额外依赖见同目录的 `requirements.txt`；脚本不会自动安装或修改环境依赖。

## 导出模型

以下为 PowerShell 命令，行尾反引号用于换行：

```powershell
python tools/onnx/export.py `
  --checkpoint models/latest_ft_d6_model/checkpoints/checkpoint-00030000 `
  --precision fp32 `
  --output-dir outputs/onnx/ft-d6-fp32-compat `
  --verify
```

导出 FP16 时改用 `--precision fp16`，并指定另一个输出目录。
DINO 默认使用 `models/dinov2` 和 `models/dinov2_vits14_pretrain.pth`，可通过
`--dinov2-repository` 和 `--dinov2-checkpoint` 修改。

深度由 checkpoint 配置决定。已训练好的 depth-6 模型不需要 `--pixel-depth`；
该参数仅用于将原版 depth-12 模型裁剪为 6 或 8 层。

输出包含：

- `dino_encoder.onnx`：DINO 特征提取模型。
- `pixel_dit.onnx`：Pixel DiT 去噪模型。
- `pixrestore-manifest.json`：模型、精度、预处理与输入输出信息。

结果保留现有浏览器管线的输入输出约定。`--output-dir` 也可指向浏览器项目的模型目录。
WebGPU 验证通过前建议另存，不覆盖旧模型。导出固定 batch size 为 1，默认分辨率为
512×512；可用 `--pixel-resolution 384` 导出固定 384×384 版本。

## 兼容替换与验证

导出时将 `x_embedder.proj1` 的 8×8、stride-8 卷积等价替换为 patch 展开和
MatMul，绕过已观察到的 WebGPU 卷积数值偏差。替换仅影响导出实例，不改变训练、
原生推理或 checkpoint。

`--verify` 比较 PyTorch 导出模块与 ONNX Runtime CUDA，不运行 WebGPU。
浏览器数值对齐步骤见 [调试文档](DEBUG.md)。

等价性测试：

```powershell
python -m unittest discover -s tools/onnx -p test_*.py
```
