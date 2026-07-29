# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.benchmark,
]

VARIANTS = {
    "480p": {
        "model": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        "model_env": "SANA_VIDEO_480P_MODEL",
        "revision": "fed3bce411c58a0f688a31afe8f52e61acc2b15f",
        "reference_class": "AutoencoderKLWan",
        "native_class": "DistributedAutoencoderKLWan",
        "dtype": torch.float32,
        "latent_shape": (1, 16, 2, 8, 12),
        "decoded_shape": (1, 3, 5, 64, 96),
        "max_abs": 1e-6,
        "relative_l2": 1e-6,
        "cosine": 0.999999,
    },
    "720p": {
        "model": "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        "model_env": "SANA_VIDEO_720P_MODEL",
        "revision": "8bda5e623d0f48cd6da3b387b10ca35d15cf1c4e",
        "reference_class": "AutoencoderKLLTX2Video",
        "native_class": "DistributedAutoencoderKLLTX2Video",
        "dtype": torch.bfloat16,
        "latent_shape": (1, 128, 2, 2, 3),
        "decoded_shape": (1, 3, 9, 64, 96),
        "max_abs": 0.02,
        "relative_l2": 0.002,
        "cosine": 0.99999,
    },
}


def _model_source(config: dict[str, Any]) -> tuple[str, str | None]:
    source = os.environ.get(config["model_env"], config["model"])
    revision = None if Path(source).is_dir() else config["revision"]
    return source, revision


def _release_cuda_model(model: torch.nn.Module) -> None:
    model.to("cpu")
    del model
    gc.collect()
    torch.accelerator.empty_cache()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "_hardware",
    [pytest.param(None, marks=hardware_marks(res={"cuda": "H100"}))],
)
def test_sana_video_vae_decode_matches_diffusers(variant: str, _hardware) -> None:
    del _hardware
    from diffusers import AutoencoderKLLTX2Video, AutoencoderKLWan

    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
        DistributedAutoencoderKLLTX2Video,
    )
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
    )

    config = VARIANTS[variant]
    reference_classes = {
        "AutoencoderKLLTX2Video": AutoencoderKLLTX2Video,
        "AutoencoderKLWan": AutoencoderKLWan,
    }
    native_classes = {
        "DistributedAutoencoderKLLTX2Video": DistributedAutoencoderKLLTX2Video,
        "DistributedAutoencoderKLWan": DistributedAutoencoderKLWan,
    }
    source, revision = _model_source(config)
    load_kwargs = {
        "subfolder": "vae",
        "torch_dtype": config["dtype"],
    }
    if revision is not None:
        load_kwargs["revision"] = revision

    generator = torch.Generator(device="cpu").manual_seed(2026)
    latent = (
        torch.randn(config["latent_shape"], generator=generator, dtype=torch.float32)
        .mul_(0.25)
        .to(device="cuda", dtype=config["dtype"])
    )

    reference_vae = reference_classes[config["reference_class"]].from_pretrained(source, **load_kwargs).eval()
    native_vae = native_classes[config["native_class"]].from_config(dict(reference_vae.config))
    native_vae.load_state_dict(reference_vae.state_dict())
    native_vae.eval()

    reference_vae.to(device="cuda", dtype=config["dtype"])
    with torch.inference_mode():
        expected = reference_vae.decode(latent, return_dict=False)[0].float().cpu()
    _release_cuda_model(reference_vae)

    native_vae.to(device="cuda", dtype=config["dtype"])
    with torch.inference_mode():
        actual = native_vae.decode(latent, return_dict=False)[0].float().cpu()
    _release_cuda_model(native_vae)

    assert expected.shape == config["decoded_shape"]
    assert actual.shape == expected.shape
    assert torch.isfinite(expected).all()
    assert torch.isfinite(actual).all()

    error = actual - expected
    max_abs = error.abs().max()
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected)
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)

    metrics = (
        f"variant={variant}, max_abs={max_abs.item():.8g}, "
        f"relative_l2={relative_l2.item():.8g}, cosine={cosine.item():.8g}"
    )
    assert max_abs.item() <= config["max_abs"], metrics
    assert relative_l2.item() <= config["relative_l2"], metrics
    assert cosine.item() >= config["cosine"], metrics
