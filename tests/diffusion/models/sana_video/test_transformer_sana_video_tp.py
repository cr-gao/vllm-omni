# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TP unit tests for SANA-Video: distributed RMSNorm and sharded attention.

CPU-only. The TP world size and the collective are mocked so a single process
simulates the shard/reduce logic a real multi-rank run depends on.
"""

import os
from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
    SanaCrossAttention,
    SanaDistributedRMSNorm,
    SanaLinearAttention,
    SanaRMSNorm,
    fused_qk_rms_norm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_MODULE = "vllm_omni.diffusion.models.sana_video.transformer_sana_video"


@pytest.fixture
def tp1_group():
    """Real single-process TP group so the parallel linear layers construct and
    run on CPU at tensor_parallel_size=1."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture
def force_default_gemm(monkeypatch):
    """Force CPU-compatible GEMM dispatch for the parallel linear layers."""
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


def test_distributed_rms_norm_tp1_matches_sana_rms_norm() -> None:
    """TP1 distributed norm must be bit-identical to the checkpoint SanaRMSNorm."""
    dim = 24
    torch.manual_seed(0)

    dense = SanaRMSNorm(dim, eps=1e-5, elementwise_affine=True)
    dense.weight.data.normal_()

    dist = SanaDistributedRMSNorm(dim, eps=1e-5)
    dist.weight.data.copy_(dense.weight.data)

    x = torch.randn(2, 5, dim)
    with patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=1):
        out = dist(x)
    ref = dense(x)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_distributed_rms_norm_tp1_bf16_weight_matches_sana_rms_norm() -> None:
    """bf16 weight must follow SanaRMSNorm's cast-to-weight-dtype branch.

    An unconditional cast back to the input dtype would diverge here; this pins
    the fp16/bf16-only cast that SanaRMSNorm performs.
    """
    dim = 24
    torch.manual_seed(0)

    dense = SanaRMSNorm(dim, eps=1e-5, elementwise_affine=True)
    dense.weight.data.normal_()
    dense.weight.data = dense.weight.data.to(torch.bfloat16)

    dist = SanaDistributedRMSNorm(dim, eps=1e-5)
    dist.weight.data = dense.weight.data.clone()

    x = torch.randn(2, 5, dim, dtype=torch.bfloat16)
    with patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=1):
        out = dist(x)
    ref = dense(x)

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_distributed_rms_norm_tp2_shards_aggregate_to_dense() -> None:
    """Two TP2 shards must reproduce dense SanaRMSNorm.

    The RMS must use the full (unsharded) hidden size: each rank reduces its
    local sum-of-squares across ranks, not a per-shard mean over its half.
    """
    full_dim, tp_size = 24, 2
    half = full_dim // tp_size
    torch.manual_seed(0)

    dense = SanaRMSNorm(full_dim, eps=1e-5, elementwise_affine=True)
    dense.weight.data.normal_()

    dist0 = SanaDistributedRMSNorm(half, eps=1e-5)
    dist1 = SanaDistributedRMSNorm(half, eps=1e-5)
    dist0.weight.data.copy_(dense.weight.data[:half])
    dist1.weight.data.copy_(dense.weight.data[half:])

    x = torch.randn(2, 5, full_dim)
    x0, x1 = x[..., :half], x[..., half:]

    # A real all-reduce returns the sum of every rank's local sum-of-squares.
    global_sum_sq = x.to(torch.float32).pow(2).sum(-1, keepdim=True)

    def fake_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        return global_sum_sq

    with (
        patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=tp_size),
        patch(f"{_MODULE}.tensor_model_parallel_all_reduce", side_effect=fake_all_reduce),
    ):
        out0 = dist0(x0)
        out1 = dist1(x1)

    out = torch.cat([out0, out1], dim=-1)
    ref = dense(x)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def _identity_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _make_qk_norms(dim: int) -> tuple[SanaDistributedRMSNorm, SanaDistributedRMSNorm]:
    """Distinct random weights so a swapped q/k slice would be caught."""
    torch.manual_seed(0)
    norm_q = SanaDistributedRMSNorm(dim, eps=1e-5)
    norm_k = SanaDistributedRMSNorm(dim, eps=1e-5)
    norm_q.weight.data.normal_()
    norm_k.weight.data.normal_()
    return norm_q, norm_k


def test_fused_qk_matches_separate_norms() -> None:
    dim = 24
    norm_q, norm_k = _make_qk_norms(dim)
    q = torch.randn(2, 5, dim)
    k = torch.randn(2, 5, dim)

    with (
        patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=2),
        patch(f"{_MODULE}.tensor_model_parallel_all_reduce", side_effect=_identity_all_reduce),
    ):
        ref_q, ref_k = norm_q(q), norm_k(k)
        fused_q, fused_k = fused_qk_rms_norm(norm_q, norm_k, q, k)

    torch.testing.assert_close(fused_q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(fused_k, ref_k, rtol=0, atol=0)


def test_fused_qk_issues_single_all_reduce_over_packed_pair() -> None:
    dim = 24
    norm_q, norm_k = _make_qk_norms(dim)
    q = torch.randn(2, 5, dim)
    k = torch.randn(2, 5, dim)

    with (
        patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=2),
        patch(
            f"{_MODULE}.tensor_model_parallel_all_reduce",
            side_effect=_identity_all_reduce,
        ) as mock_ar,
    ):
        fused_qk_rms_norm(norm_q, norm_k, q, k)

    assert mock_ar.call_count == 1
    packed = mock_ar.call_args[0][0]
    assert packed.shape[-1] == 2


def test_distributed_rms_norm_weight_loader_shards_dim0() -> None:
    """Each rank loads its own dim-0 slice of the global affine weight."""
    full_dim, tp_size, rank = 24, 2, 1
    half = full_dim // tp_size

    norm = SanaDistributedRMSNorm(half, eps=1e-5)
    full_weight = torch.randn(full_dim)

    with (
        patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=tp_size),
        patch(f"{_MODULE}.get_tensor_model_parallel_rank", return_value=rank),
    ):
        norm.weight.weight_loader(norm.weight, full_weight)

    torch.testing.assert_close(norm.weight.data, full_weight[rank * half : (rank + 1) * half])


def test_distributed_rms_norm_weight_loader_tp1_copies_full() -> None:
    dim = 24
    norm = SanaDistributedRMSNorm(dim, eps=1e-5)
    full_weight = torch.randn(dim)

    norm.weight.weight_loader(norm.weight, full_weight)

    torch.testing.assert_close(norm.weight.data, full_weight)


_SELF_ATTN_GOLDEN = torch.tensor(
    [
        5.0944157,
        -16.9981823,
        55.4457436,
        42.5174408,
        -9.3055859,
        19.3744793,
        -47.4345360,
        -5.8110709,
        -14.0679598,
        -22.3440742,
        -8.3834534,
        8.9109459,
    ]
)


def _fill_by_name(module: torch.nn.Module) -> None:
    """Deterministic per-parameter fill that is stable across TP refactors."""
    torch.manual_seed(0)
    for _, param in sorted(module.named_parameters()):
        param.data.normal_()


def _self_attn_inputs(seq_len: int, dim: int, head_dim: int):
    torch.manual_seed(1)
    x = torch.randn(2, seq_len, dim)
    freqs_cos = torch.randn(1, seq_len, 1, head_dim)
    freqs_sin = torch.randn(1, seq_len, 1, head_dim)
    return x, (freqs_cos, freqs_sin)


def test_self_attention_tp1_matches_golden_and_keeps_param_names(tp1_group, force_default_gemm) -> None:
    """TP1 self-attention stays bit-identical to PR1 and preserves checkpoint keys."""
    dim, num_heads, head_dim, seq_len = 24, 4, 6, 8
    attn = SanaLinearAttention(
        dim=dim, num_heads=num_heads, head_dim=head_dim, dropout=0.0, bias=False, qk_norm="rms_norm_across_heads"
    )
    attn.eval()
    _fill_by_name(attn)
    x, rotary_emb = _self_attn_inputs(seq_len, dim, head_dim)

    with torch.no_grad():
        out = attn(x, rotary_emb=rotary_emb)

    assert {name for name, _ in attn.named_parameters()} == {
        "to_q.weight",
        "to_k.weight",
        "to_v.weight",
        "norm_q.weight",
        "norm_k.weight",
        "to_out.0.weight",
        "to_out.0.bias",
    }
    torch.testing.assert_close(out.flatten()[:12], _SELF_ATTN_GOLDEN, rtol=1e-5, atol=1e-5)


def _mock_tp2(mocker):
    """Mock a TP world size of 2 for module construction (no real group)."""
    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=2)
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        return_value=2,
    )
    tp_group = mocker.MagicMock()
    tp_group.world_size = 2
    mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp_group)


def test_self_attention_tp2_shards_projections_and_heads(mocker) -> None:
    """TP2 splits heads column-wise (q/k/v) and the output projection row-wise."""
    dim, num_heads, head_dim = 24, 4, 6
    inner_dim = num_heads * head_dim
    _mock_tp2(mocker)

    attn = SanaLinearAttention(
        dim=dim, num_heads=num_heads, head_dim=head_dim, dropout=0.0, bias=False, qk_norm="rms_norm_across_heads"
    )

    assert attn.heads == num_heads // 2
    assert attn.to_q.weight.shape == (inner_dim // 2, dim)
    assert attn.to_k.weight.shape == (inner_dim // 2, dim)
    assert attn.to_v.weight.shape == (inner_dim // 2, dim)
    assert attn.to_out[0].weight.shape == (dim, inner_dim // 2)
    assert attn.norm_q.weight.shape == (inner_dim // 2,)
    assert attn.norm_k.weight.shape == (inner_dim // 2,)


def test_cross_attention_tp2_shards_projections_and_heads(mocker) -> None:
    """TP2 cross-attention shards q/k/v column-wise, to_out row-wise, heads locally."""
    dim, cross_dim, num_heads, head_dim = 24, 16, 4, 6
    inner_dim = num_heads * head_dim
    _mock_tp2(mocker)

    attn = SanaCrossAttention(
        dim=dim,
        cross_attention_dim=cross_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        dropout=0.0,
        bias=False,
        out_bias=True,
        qk_norm="rms_norm_across_heads",
        prefix="attn2",
    )

    assert attn.heads == num_heads // 2
    assert attn.to_q.weight.shape == (inner_dim // 2, dim)
    assert attn.to_k.weight.shape == (inner_dim // 2, cross_dim)
    assert attn.to_v.weight.shape == (inner_dim // 2, cross_dim)
    assert attn.to_out[0].weight.shape == (dim, inner_dim // 2)
    assert isinstance(attn.norm_q, SanaDistributedRMSNorm)
    assert attn.norm_q.weight.shape == (inner_dim // 2,)


_TINY_TRANSFORMER_CONFIG = {
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


def test_tp2_load_weights_shards_full_checkpoint_without_missing(mocker) -> None:
    """The full checkpoint loads under TP2 via weight_loaders with no missing/unexpected keys."""
    from diffusers import SanaVideoTransformer3DModel as DiffusersTransformer

    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel

    _mock_tp2(mocker)
    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_rank", return_value=0)
    mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0)

    reference = DiffusersTransformer(**_TINY_TRANSFORMER_CONFIG)
    model = SanaVideoTransformer3DModel(**_TINY_TRANSFORMER_CONFIG)

    loaded = model.load_weights(list(reference.state_dict().items()))

    assert loaded == set(model.state_dict())
