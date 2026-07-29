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


@pytest.mark.parametrize(
    ("elementwise_affine", "bias", "expected_state_keys"),
    [
        (False, False, set()),
        (False, True, set()),
        (True, False, {"weight"}),
        (True, True, {"weight", "bias"}),
    ],
)
@pytest.mark.parametrize(
    ("input_dtype", "parameter_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
    ],
)
def test_sana_rms_norm_matches_diffusers(
    elementwise_affine,
    bias,
    expected_state_keys,
    input_dtype,
    parameter_dtype,
):
    from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm

    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import SanaRMSNorm

    reference = DiffusersRMSNorm(8, eps=1e-5, elementwise_affine=elementwise_affine, bias=bias).to(
        dtype=parameter_dtype
    )
    actual = SanaRMSNorm(8, eps=1e-5, elementwise_affine=elementwise_affine, bias=bias).to(dtype=parameter_dtype)

    assert actual.eps == reference.eps
    assert actual.elementwise_affine == reference.elementwise_affine
    assert actual.dim == reference.dim
    assert set(actual.state_dict()) == expected_state_keys
    assert set(actual.state_dict()) == set(reference.state_dict())

    if elementwise_affine:
        weight = torch.linspace(0.5, 1.5, 8, dtype=parameter_dtype)
        actual.weight.data.copy_(weight)
        reference.weight.data.copy_(weight)
        if bias:
            norm_bias = torch.linspace(-0.25, 0.25, 8, dtype=parameter_dtype)
            actual.bias.data.copy_(norm_bias)
            reference.bias.data.copy_(norm_bias)
    else:
        assert actual.weight is None
        assert actual.bias is None

    hidden_states = torch.linspace(-2.0, 2.0, 48, dtype=input_dtype).reshape(2, 3, 8)
    expected = reference(hidden_states)
    result = actual(hidden_states)

    assert result.dtype == expected.dtype
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("dim", "max_seq_len", "theta"), [(4, 8, 10000.0), (12, 32, 256.0)])
def test_native_rope_matches_diffusers(dim, max_seq_len, theta, freqs_dtype):
    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
        _get_1d_rotary_pos_embed,
    )

    expected_cos, expected_sin = get_1d_rotary_pos_embed(
        dim,
        max_seq_len,
        theta,
        use_real=True,
        repeat_interleave_real=True,
        freqs_dtype=freqs_dtype,
    )
    actual_cos, actual_sin = _get_1d_rotary_pos_embed(dim, max_seq_len, theta, freqs_dtype)

    torch.testing.assert_close(actual_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(actual_sin, expected_sin, rtol=0, atol=0)


def test_tiny_transformer_matches_diffusers_and_frozen_output():
    from diffusers import SanaVideoTransformer3DModel as DiffusersTransformer
    from diffusers.configuration_utils import ConfigMixin
    from diffusers.models.modeling_utils import ModelMixin

    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
        SanaLinearAttention,
        SanaRMSNorm,
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
    assert isinstance(block.attn1.norm_q, SanaRMSNorm)
    assert isinstance(block.attn1.norm_k, SanaRMSNorm)
    assert isinstance(block.attn2.norm_q, SanaRMSNorm)
    assert isinstance(block.attn2.norm_k, SanaRMSNorm)
    assert isinstance(model.caption_norm, SanaRMSNorm)

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
