# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_TINY_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "num_attention_heads": 2,
    "attention_head_dim": 12,
    "num_layers": 1,
    "num_cross_attention_heads": 2,
    "cross_attention_head_dim": 12,
    "cross_attention_dim": 24,
    "caption_channels": 8,
    "mlp_ratio": 2.0,
    "patch_size": (1, 2, 2),
    "sample_size": 4,
    "rope_max_seq_len": 32,
}

_GOLDEN_PREFIX = torch.tensor(
    [
        -0.1159307063,
        0.2003862262,
        -0.3754127920,
        -0.2403523028,
        -1.2376221418,
        -0.6513092518,
        -0.3358457983,
        -0.9894289970,
        0.7858567238,
        0.5089095831,
        -0.5362522602,
        -0.3732060790,
    ]
)


def test_tiny_transformer_matches_diffusers_and_frozen_output():
    from diffusers import SanaVideoTransformer3DModel as DiffusersTransformer
    from diffusers.configuration_utils import ConfigMixin
    from diffusers.models.modeling_utils import ModelMixin

    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
        SanaLinearAttention,
        SanaVideoTransformerConfig,
        SanaVideoTransformerOutput,
    )

    torch.manual_seed(7)
    reference = DiffusersTransformer(**_TINY_CONFIG).eval()
    model = SanaVideoTransformer3DModel(**_TINY_CONFIG).eval()
    assert not isinstance(model, ModelMixin)
    assert not isinstance(model, ConfigMixin)
    assert isinstance(model.config, SanaVideoTransformerConfig)
    assert set(model.state_dict()) == set(reference.state_dict())
    model.load_state_dict(reference.state_dict())

    block = model.transformer_blocks[0]
    assert isinstance(block.attn1, SanaLinearAttention)
    assert isinstance(block.attn2.attn, Attention)
    assert block.attn2.attn.role == "cross"

    torch.manual_seed(11)
    hidden_states = torch.randn(1, 4, 3, 4, 4)
    encoder_hidden_states = torch.randn(1, 5, 8)
    encoder_attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    timestep = torch.tensor([500.0])

    with torch.no_grad():
        expected = reference(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask=encoder_attention_mask,
        ).sample
        actual_output = model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask=encoder_attention_mask,
        )
        actual = actual_output.sample

    assert isinstance(actual_output, SanaVideoTransformerOutput)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.flatten()[: len(_GOLDEN_PREFIX)], _GOLDEN_PREFIX, rtol=1e-5, atol=1e-5)


def test_native_transformer_config_filters_diffusers_metadata():
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import SanaVideoTransformerConfig

    config = SanaVideoTransformerConfig.from_dict(
        _TINY_CONFIG | {"_class_name": "SanaVideoTransformer3DModel", "_diffusers_version": "0.38.0"}
    )
    assert config.patch_size == (1, 2, 2)

    with pytest.raises(ValueError, match="unsupported_field"):
        SanaVideoTransformerConfig.from_dict(_TINY_CONFIG | {"unsupported_field": True})


def test_dual_vae_variant_resolution():
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
        DistributedAutoencoderKLLTX2Video,
    )
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
    )
    from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
    from vllm_omni.diffusion.models.sana_video.pipeline_sana_video import (
        SanaVideoPipeline,
        _resolve_vae_class_and_dtype,
    )

    assert not issubclass(SanaVideoPipeline, DiffusionPipeline)
    assert issubclass(SanaVideoPipeline, ProgressBarMixin)

    vae_class, dtype = _resolve_vae_class_and_dtype("AutoencoderKLWan", torch.bfloat16)
    assert vae_class is DistributedAutoencoderKLWan
    assert dtype is torch.float32

    vae_class, dtype = _resolve_vae_class_and_dtype("AutoencoderKLLTX2Video", torch.bfloat16)
    assert vae_class is DistributedAutoencoderKLLTX2Video
    assert dtype is torch.bfloat16

    with pytest.raises(ValueError, match="Unsupported SANA-Video VAE"):
        _resolve_vae_class_and_dtype("UnknownVAE", torch.bfloat16)
