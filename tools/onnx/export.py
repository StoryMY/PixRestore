"""Export fixed 512x512 PixRestore-S components for ONNX Runtime WebGPU.

The exported graphs deliberately stop at the Pixel DiT denoiser. The browser
constructs the one-step sampler input and applies the published one-step update;
keeping random noise generation outside ONNX makes the browser path reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import torch
from torch import nn

# DINO's optional xFormers kernels cannot be traced by the legacy ONNX exporter.
os.environ['XFORMERS_DISABLED'] = '1'


OPSET = 17
PIXRESTORE_S_DEPTH_8_BLOCKS = (0, 1, 3, 4, 6, 7, 9, 11)
PIXRESTORE_S_DEPTH_6_BLOCKS = (0, 2, 4, 6, 8, 11)


class DinoLayerEncoder(nn.Module):
    """Expose exactly the DINOv2 intermediate layers consumed by PixRestore-S."""

    def __init__(self, encoder: nn.Module, layer_ids: list[int]):
        super().__init__()
        self.encoder = encoder
        self.layer_ids = layer_ids

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features, final = self.encoder.forward_with_features(image)
        features[max(features)] = final
        return tuple(features[layer] for layer in self.layer_ids)


class PixelDitDenoiser(nn.Module):
    """Expose Pixel DiT as a pure denoiser with explicit DINO feature inputs."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.use_fp16_autocast = next(model.parameters()).dtype == torch.float16

    def forward(
        self,
        pixel_input: torch.Tensor,
        timestep: torch.Tensor,
        *dino_features: torch.Tensor,
    ) -> torch.Tensor:
        # PixRestore inference runs inside Accelerator autocast. TimestepEmbedder
        # intentionally creates float32 sinusoidal values, so FP16 model weights
        # need the same CUDA autocast boundary during export and verification.
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16_autocast):
            prediction, _ = self.model(pixel_input, timestep, z=list(dino_features))
        return prediction


def workspace_default_pixrestore_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    pixrestore_root = workspace_default_pixrestore_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixrestore-root", type=Path, default=pixrestore_root)
    parser.add_argument("--checkpoint", type=Path, default=pixrestore_root / "models" / "pixrestore-s")
    parser.add_argument("--config", type=Path, help="Optional PixRestore-S config.json override.")
    parser.add_argument("--dinov2-repository", type=Path, default=pixrestore_root / "models" / "dinov2")
    parser.add_argument("--dinov2-checkpoint", type=Path, default=pixrestore_root / "models" / "dinov2_vits14_pretrain.pth")
    parser.add_argument("--output-dir", type=Path, help="Destination directory. Defaults to a variant-specific directory when pruning or changing resolution.")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--pixel-depth", type=int, choices=(6, 8), help="Export a supported pruned Pixel DiT variant from the released 12-block model.")
    parser.add_argument("--pixel-resolution", type=int, choices=(384,), help="Export Pixel DiT at a fixed lower resolution while retaining the 448x448 DINO input.")
    parser.add_argument("--verify", action="store_true", help="Compare both exported components with CUDA ONNX Runtime.")
    parser.add_argument("--dino-atol", type=float, default=1e-2, help="DINO absolute tolerance for CUDA ORT verification.")
    parser.add_argument("--dino-rtol", type=float, default=2e-2, help="DINO relative tolerance for CUDA ORT verification.")
    parser.add_argument("--pixel-atol", type=float, default=5e-3, help="Pixel DiT absolute tolerance for CUDA ORT verification.")
    parser.add_argument("--pixel-rtol", type=float, default=2e-2, help="Pixel DiT relative tolerance for CUDA ORT verification.")
    return parser.parse_args()


def import_pixrestore(root: Path):
    root = root.expanduser().resolve()
    if not (root / "inference.py").is_file():
        raise FileNotFoundError(f"PixRestore source root does not contain inference.py: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import inference
    from pixrestore.vision import IMAGENET_MEAN, IMAGENET_STD, load_dinov2, preprocess_dinov2

    return inference, load_dinov2, preprocess_dinov2, IMAGENET_MEAN, IMAGENET_STD


def disable_pixrestore_torch_compile() -> None:
    """Restore PixRestore functions before legacy TorchScript ONNX tracing.

    torch.compile and torch.jit.trace cannot be nested. The exporter uses the
    legacy tracer because the current model graph is not dynamo-exportable.
    """
    from pixrestore.models import layers, swiglu

    def unwrap(function):
        return getattr(function, "__wrapped__", getattr(function, "_torchdynamo_orig_callable", function))

    layers.modulate_adasin = unwrap(layers.modulate_adasin)
    layers.modulate = unwrap(layers.modulate)
    layers.LightningDiTBlock.forward = unwrap(layers.LightningDiTBlock.forward)
    layers.FinalLayer.forward = unwrap(layers.FinalLayer.forward)
    swiglu.SwiGLUFFN.forward = unwrap(swiglu.SwiGLUFFN.forward)


def prune_pixel_dit_depth(model: nn.Module, target_depth: int | None) -> tuple[int, tuple[int, ...]]:
    """Prune the released 12-block Pixel DiT while preserving its per-block gates."""
    source_depth = len(model.blocks)
    if target_depth is None:
        return source_depth, tuple(range(source_depth))
    if source_depth != 12 or target_depth not in (6, 8):
        raise ValueError(f"Only released 12-to-6 and 12-to-8 Pixel DiT pruning are supported; got {source_depth}-to-{target_depth}.")

    kept_blocks = PIXRESTORE_S_DEPTH_6_BLOCKS if target_depth == 6 else PIXRESTORE_S_DEPTH_8_BLOCKS
    model.blocks = nn.ModuleList([model.blocks[index] for index in kept_blocks])
    model.depth = len(kept_blocks)

    gate_block_embed = getattr(model, "gate_block_embed", None)
    if gate_block_embed is not None:
        if gate_block_embed.num_embeddings != source_depth:
            raise ValueError("Pixel DiT gate_block_embed does not match the source block depth.")
        pruned_embedding = nn.Embedding(len(kept_blocks), gate_block_embed.embedding_dim)
        with torch.no_grad():
            pruned_embedding.weight.copy_(gate_block_embed.weight[list(kept_blocks)])
        model.gate_block_embed = pruned_embedding

    return source_depth, kept_blocks


def resolve_optional_file(path: Path | None, label: str) -> str | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def resolve_repository(path: Path | None) -> str | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"DINOv2 repository not found: {path}")
    return str(path)


def export_onnx(
    module: nn.Module,
    arguments: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    torch.onnx.export(
        module,
        arguments,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET,
        dynamo=False,
        external_data=True,
        do_constant_folding=True,
    )
    onnx.checker.check_model(output_path)


def error_stats(reference: torch.Tensor, actual: np.ndarray, atol: float, rtol: float) -> tuple[float, float, float, int]:
    reference_array = reference.detach().float().cpu().numpy()
    difference = np.abs(reference_array - actual.astype(np.float32))
    max_abs = float(difference.max())
    mean_abs = float(difference.mean())
    reference_scale = max(float(np.abs(reference_array).max()), np.finfo(np.float32).tiny)
    allowed_difference = atol + rtol * np.abs(reference_array)
    failures = int(np.count_nonzero(difference > allowed_difference))
    return max_abs, mean_abs, max_abs / reference_scale, failures


def verify_models(
    output_dir: Path,
    dino_image: torch.Tensor,
    pixel_input: torch.Tensor,
    timestep: torch.Tensor,
    dino_encoder: DinoLayerEncoder,
    pixel_dit: PixelDitDenoiser,
    dino_input_name: str,
    dino_output_names: list[str],
    pixel_input_names: list[str],
    dino_atol: float,
    dino_rtol: float,
    pixel_atol: float,
    pixel_rtol: float,
) -> None:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Install tools/onnx/requirements.txt before using --verify.") from error

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is required for --verify.")
    ort.preload_dlls()
    providers = ["CUDAExecutionProvider"]

    with torch.inference_mode():
        dino_reference = dino_encoder(dino_image)
        pixel_reference = pixel_dit(pixel_input, timestep, *dino_reference)

    dino_session = ort.InferenceSession(str(output_dir / "dino_encoder.onnx"), providers=providers)
    if dino_session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("DINO CUDA provider failed to initialize")
    dino_actual = dino_session.run(None, {dino_input_name: dino_image.detach().cpu().numpy()})
    dino_errors = [
        error_stats(reference, actual, dino_atol, dino_rtol)
        for reference, actual in zip(dino_reference, dino_actual, strict=True)
    ]

    pixel_session = ort.InferenceSession(str(output_dir / "pixel_dit.onnx"), providers=providers)
    if pixel_session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("Pixel DiT CUDA provider failed to initialize")
    pixel_feeds = {
        pixel_input_names[0]: pixel_input.detach().cpu().numpy(),
        pixel_input_names[1]: timestep.detach().cpu().numpy(),
    }
    pixel_feeds.update({name: value for name, value in zip(pixel_input_names[2:], dino_actual, strict=True)})
    pixel_actual = pixel_session.run(None, pixel_feeds)[0]
    pixel_error = error_stats(pixel_reference, pixel_actual, pixel_atol, pixel_rtol)

    for name, (max_abs, mean_abs, relative_max, failures) in zip(dino_output_names, dino_errors, strict=True):
        print(f"{name}: max abs {max_abs:.6g}, mean abs {mean_abs:.6g}, max/reference-max {relative_max:.6g}, tolerance failures {failures}")
    pixel_max_abs, pixel_mean_abs, pixel_relative_max, pixel_failures = pixel_error
    largest_error = max(*(error[0] for error in dino_errors), pixel_max_abs)
    print(f"pixel_dit: max abs {pixel_max_abs:.6g}, mean abs {pixel_mean_abs:.6g}, max/reference-max {pixel_relative_max:.6g}, tolerance failures {pixel_failures}")
    dino_failures = sum(error[3] for error in dino_errors)
    if dino_failures or pixel_failures:
        raise RuntimeError(
            f"CUDA ONNX Runtime verification failed: {dino_failures} DINO and {pixel_failures} Pixel DiT values exceed tolerance; "
            f"largest absolute error was {largest_error:.6g}."
        )


def main() -> None:
    args = parse_args()
    if min(args.dino_atol, args.dino_rtol, args.pixel_atol, args.pixel_rtol) <= 0:
        raise ValueError("All verification tolerances must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("PixRestore-S export requires CUDA because the model is too large for CPU export.")
    if args.output_dir is None:
        suffixes = [f"depth{args.pixel_depth}" if args.pixel_depth else None, str(args.pixel_resolution) if args.pixel_resolution else None]
        suffix = "-".join(suffix for suffix in suffixes if suffix)
        args.output_dir = args.pixrestore_root / Path(f"outputs/onnx/pixrestore-s{'-' + suffix if suffix else ''}")

    inference, load_dinov2, preprocess_dinov2, imagenet_mean, imagenet_std = import_pixrestore(args.pixrestore_root)
    disable_pixrestore_torch_compile()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = inference.find_config_path(checkpoint, str(args.config) if args.config else None)
    config_data = inference.read_config(config_path)
    weight_path = inference.find_weights(checkpoint)
    state = inference.read_state_dict(weight_path)
    config_data = inference.apply_checkpoint_architecture_flags(config_data, state)
    config = SimpleNamespace(**config_data)
    training_resolution = int(config.resolution)
    if training_resolution != 512:
        raise ValueError(f"PixRestore-S browser exports require a model trained at 512x512, but config resolution is {training_resolution}.")
    resolution = args.pixel_resolution or training_resolution
    if not bool(config.use_venc):
        raise ValueError("PixRestore-S browser v1 requires DINOv2 conditioning (config.use_venc=true).")

    device = torch.device("cuda")
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    model = inference.build_model(config)
    inference.load_weights(model, weight_path, state=state)
    source_depth, kept_blocks = prune_pixel_dit_depth(model, args.pixel_depth)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    # ONNX Runtime WebGPU needs explicit MatMul/Softmax rather than PyTorch fused SDPA.
    model.disable_fused_attn()
    from pixrestore.onnx_compat import apply_webgpu_compat
    apply_webgpu_compat(model)

    repository = resolve_repository(args.dinov2_repository)
    dino_checkpoint = resolve_optional_file(args.dinov2_checkpoint, "DINOv2 checkpoint")
    encoder = load_dinov2(config.encoder_type, device, repository=repository, checkpoint=dino_checkpoint)
    encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)

    layer_ids = [int(layer) for layer in config.encoder_layers]
    dino_encoder = DinoLayerEncoder(encoder, layer_ids).to(device).eval()
    pixel_dit = PixelDitDenoiser(model).to(device).eval()
    torch.manual_seed(42)
    lq = torch.rand((1, 3, resolution, resolution), device=device, dtype=dtype) * 2 - 1
    pixel_input = torch.randn((1, 6, resolution, resolution), device=device, dtype=dtype)
    pixel_input[:, :3] = lq
    timestep = torch.ones((1,), device=device, dtype=torch.float32)
    with torch.inference_mode():
        dino_image = preprocess_dinov2(lq, int(config.encoder_input_size))
        dino_features = dino_encoder(dino_image)
        pixel_dit(pixel_input, timestep, *dino_features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dino_output_names = [f"dino_layer_{layer}" for layer in layer_ids]
    pixel_input_names = ["pixel_input", "timestep", *dino_output_names]
    export_onnx(
        dino_encoder,
        (dino_image,),
        args.output_dir / "dino_encoder.onnx",
        ["image"],
        dino_output_names,
    )
    export_onnx(
        pixel_dit,
        (pixel_input, timestep, *dino_features),
        args.output_dir / "pixel_dit.onnx",
        pixel_input_names,
        ["x_pred"],
    )

    manifest = {
        "format": "pixrestore-webgpu-v1",
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "precision": args.precision,
        "resolution": resolution,
        "models": {"dinoEncoder": "dino_encoder.onnx", "pixelDit": "pixel_dit.onnx"},
        "dino": {
            "encoderType": config.encoder_type,
            "inputSize": int(config.encoder_input_size),
            "input": {
                "name": "image",
                "shape": list(dino_image.shape),
                "preprocessing": {
                    "sourceRange": [-1, 1],
                    "resize": {"mode": "bicubic", "antialias": True},
                    "pixelRange": [0, 1],
                    "normalization": {"mean": list(imagenet_mean), "std": list(imagenet_std)},
                },
            },
            "layerIds": layer_ids,
            "outputs": dino_output_names,
        },
        "pixelDit": {
            "input": {"name": "pixel_input", "shape": [1, 6, resolution, resolution]},
            "timestep": {"name": "timestep", "shape": [1], "value": 1.0},
            "output": "x_pred",
            "depth": {"source": source_depth, "keptBlocks": list(kept_blocks)},
            "sourceResolution": training_resolution,
            "patchEmbedding": "reshape-transpose-matmul",
        },
        "singleStepSampling": {
            "output": "x_pred",
            "noiseInput": "standard_normal_like_lq",
            "lqConditionStrength": float(getattr(config, "cond_strength_aelq_test", 0.0)),
            "dinoConditionStrength": float(getattr(config, "cond_strength_venc", 0.0)),
            "usesAelq": bool(getattr(config, "use_aelq", False)),
        },
    }
    (args.output_dir / "pixrestore-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported fixed {resolution}x{resolution} PixRestore-S assets to {args.output_dir}")

    if args.verify:
        verify_models(
            args.output_dir,
            dino_image,
            pixel_input,
            timestep,
            dino_encoder,
            pixel_dit,
            "image",
            dino_output_names,
            pixel_input_names,
            args.dino_atol,
            args.dino_rtol,
            args.pixel_atol,
            args.pixel_rtol,
        )


if __name__ == "__main__":
    main()
