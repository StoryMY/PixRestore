# 固定输入对齐验证

生成固定输入和 CUDA 参考：

```powershell
python tools/onnx/alignment.py export `
  --model-dir outputs/onnx/ft-d6-fp32-compat `
  --image path/to/input.png `
  --output-dir outputs/onnx/alignment-compat
```

让浏览器加载同一套 ONNX 文件，在对齐验证界面选择生成的 `alignment-inputs.json`。
运行后下载原始结果，再执行：

```powershell
python tools/onnx/alignment.py compare `
  --reference outputs/onnx/alignment-compat/alignment-cuda.json `
  --actual path/to/alignment-webgpu.json `
  --output-dir outputs/onnx/alignment-compat/comparison
```

将 `--actual` 替换为实际下载路径。脚本检查模型哈希、输入哈希和图优化设置，输出
原始张量误差及 CUDA、WebGPU、放大差异的横向对比图。

关闭图优化时，用原有输入另生成一份参考：

```powershell
python tools/onnx/alignment.py reference `
  --model-dir outputs/onnx/ft-d6-fp32-compat `
  --inputs outputs/onnx/alignment-compat/alignment-inputs.json `
  --graph-optimization disabled `
  --output outputs/onnx/alignment-compat/alignment-cuda-no-opt.json
```

浏览器也需勾选关闭图优化，比较时使用对应的新参考和下载结果。
`reference` 复用文件中的输入及固定 DINO 特征，不重新生成噪声。

