#!/usr/bin/env python
"""Fast, self-contained checks for the standard and DINO-GAN training paths."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pixrestore import LightningDiT_PixelDiffusion, MeanFlowIR_PixelDiffusion
from pixrestore.data import PairedJsonlDataset, collate_pairs
from pixrestore.gan import MultiLayerDinoDiscriminator, as_feature_list, set_requires_grad
from train import load_pretrained, read_evaluation_manifest


def check_pretrained_block_selection() -> None:
    def make_model(depth: int, width: int = 2):
        model = nn.Module()
        model.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        model.gate_block_embed = nn.Embedding(depth, width)
        model.final_layer = nn.Linear(width, 3)
        return model

    source = make_model(12)
    with torch.no_grad():
        for index, block in enumerate(source.blocks):
            block.weight.fill_(index + 1)
            block.bias.fill_(-index - 1)
            source.gate_block_embed.weight[index].fill_(index + 20)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "source.pth"
        torch.save(source.state_dict(), path)
        for indices in ([0, 2, 4, 6, 8, 11], [0, 1, 3, 4, 6, 7, 9, 11]):
            model, ema = make_model(len(indices)), make_model(len(indices))
            load_pretrained(model, ema, path, block_indices=indices)
            for target, original in enumerate(indices):
                torch.testing.assert_close(model.blocks[target].weight, source.blocks[original].weight)
                torch.testing.assert_close(model.blocks[target].bias, source.blocks[original].bias)
            torch.testing.assert_close(model.gate_block_embed.weight, source.gate_block_embed.weight[indices])
            torch.testing.assert_close(model.final_layer.weight, source.final_layer.weight)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(ema.state_dict()[key], value)
            small_path = Path(directory) / "small.pth"
            torch.save(ema.state_dict(), small_path)
            reloaded = make_model(len(indices))
            load_pretrained(reloaded, make_model(len(indices)), small_path)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(reloaded.state_dict()[key], value)

        cases = [
            (path, [0, 2], 2),
            (path, [0, 2, 4, 6, 8, 8], 2),
            (path, [0, 2, 4, 6, 8, 12], 2),
            (path, [0, 2, 4, 6, 8, -1], 2),
            (path, [0, 2, 4, 6, 8, True], 2),
            (path, [0, 2, 4, 6, 8, 11], 3),
            (small_path, list(range(8)), 2),
        ]
        for checkpoint, indices, width in cases:
            depth = 8 if checkpoint == small_path else 6
            try:
                load_pretrained(make_model(depth, width), make_model(depth, width), checkpoint, block_indices=indices)
            except ValueError:
                pass
            else:
                raise RuntimeError(f"Invalid block selection accepted: {indices}, width={width}")
    print("OK: depth-6/depth-8 initialization, EMA, reload and invalid selections")


class TinyFrozenEncoder(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, feature_dim, bias=False)
        self.num_layers = num_layers
        self.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        tokens = F.adaptive_avg_pool2d(image, (2, 2)).flatten(2).transpose(1, 2)
        tokens = self.projection(tokens)
        return [tokens * (1 + 0.1 * index) for index in range(self.num_layers)]


def make_batch() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        Image.new("RGB", (24, 20), (64, 96, 128)).save(root / "lq.png")
        Image.new("RGB", (24, 20), (96, 128, 160)).save(root / "gt.png")
        manifest = root / "pairs.jsonl"
        manifest.write_text(
            json.dumps({"lq": str(root / "lq.png"), "gt": str(root / "gt.png")}) + "\n"
        )
        dataset = PairedJsonlDataset([manifest], crop_size=16)
        return next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate_pairs)))


def check_evaluation_manifest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "nested").mkdir()
        Image.new("RGB", (24, 20)).save(root / "nested" / "lq.png")
        Image.new("RGB", (24, 20)).save(root / "nested" / "gt.png")
        manifest = root / "pairs.jsonl"
        manifest.write_text(
            json.dumps({"lq": "nested/lq.png", "gt": "nested/gt.png"}) + "\n"
        )
        samples = read_evaluation_manifest(manifest)
        if samples != [(root / "nested" / "lq.png", root / "nested" / "gt.png", "000001_lq")]:
            raise RuntimeError(f"Unexpected evaluation manifest samples: {samples}")


def make_config(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        use_time_shift=False,
        time_shift_base_dimension=4096,
        cond_strength_aelq_list=[5.0, 1.0],
        aug_noise=False,
        use_aelq=True,
        use_venc=True,
        weak_cond_strength_aelq_list=[0.99, 1.0],
        weak_cond_strength_venc=0.0,
        cond_strength_venc=1.0,
        cond_strength_aelq_test=1.0,
        t_eps=0.05,
        fixed_train_t=1.0 if mode == "gan" else None,
        dino_layer_router_mode="content",
        use_dino_layer_router=True,
        dino_layer_router_loss_weight=0.5,
        use_hierar_loss=True,
        dino_hierar_loss_type="cosine",
        dino_hierar_loss_weight=0.5,
        average_hierar_loss_weights=False,
        same_condition_layer_weights=False,
        gate_temperature=1.0,
    )


def run(mode: str) -> None:
    torch.manual_seed(0)
    config = make_config(mode)
    feature_dim = 24
    num_layers = 3
    encoder = TinyFrozenEncoder(feature_dim, num_layers)
    model = LightningDiT_PixelDiffusion(
        input_size=16,
        patch_size=8,
        in_channels=6,
        out_channels=3,
        hidden_size=48,
        depth=2,
        num_heads=6,
        mlp_ratio=2,
        z_dims=feature_dim,
        encdim_ratio=2,
        num_fused_layers=num_layers,
        pca_dim=24,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        use_dino_layer_router=True,
        dino_layer_router_mode="content",
        same_condition_layer_weights=False,
        gate_temperature=1.0,
        feature_norm="channel_rmsnorm",
    )
    flow = MeanFlowIR_PixelDiffusion(
        accelerator=SimpleNamespace(device=torch.device("cpu")),
        args=config,
        cfg_ratio=0.1,
    )
    batch = make_batch()
    lq, hq = batch["lq"], batch["hq"]
    lq_features = encoder(lq)
    hq_features = encoder(hq)
    fake_features = None

    def feature_loss(image):
        nonlocal fake_features
        fake_features = encoder(image)
        return fake_features

    loss, backward_loss, _ = flow.loss_fm(
        model,
        lq,
        hq,
        lq_features,
        z_hq=hq_features,
        dino_loss_fn=feature_loss,
    )

    discriminator = None
    if mode == "gan":
        discriminator = MultiLayerDinoDiscriminator(feature_dim, num_layers)
        set_requires_grad(discriminator, False)
        generator_gan_loss = discriminator(fake_features, real=True) * 0.5
        backward_loss = backward_loss + generator_gan_loss

    backward_loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Model received no gradients")

    if discriminator is not None:
        set_requires_grad(discriminator, True)
        discriminator.zero_grad(set_to_none=True)
        fake_loss = discriminator(
            [feature.detach() for feature in as_feature_list(fake_features)], real=False
        )
        real_loss = discriminator(
            [feature.detach() for feature in as_feature_list(hq_features)], real=True
        )
        (fake_loss + real_loss).backward()
        if not any(parameter.grad is not None for parameter in discriminator.parameters()):
            raise RuntimeError("Discriminator received no gradients")

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss}")
    print(f"OK: {mode} training path, loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("standard", "gan"), required=True)
    check_evaluation_manifest()
    check_pretrained_block_selection()
    run(parser.parse_args().mode)
