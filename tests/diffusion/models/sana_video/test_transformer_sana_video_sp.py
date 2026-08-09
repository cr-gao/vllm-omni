# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SP unit tests for SANA-Video: balanced frame split sizes, linear-attention
state reduction and the uneven frame gather.

CPU-only. The SP world size, rank and the collectives are mocked so a single
process simulates the shard/reduce logic a real multi-rank run depends on.
"""

import os

import pytest
import torch

from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
    SanaLinearAttention,
    _sp_frame_split_sizes,
    _sp_gather_frames,
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


# ── frame split sizes ──


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("num_frames", [5, 6, 9, 11, 21])
def test_frame_split_sizes_are_balanced_and_cover_all_frames(num_frames, world_size) -> None:
    """Exactly world_size non-empty near-equal chunks that sum to num_frames.

    torch.chunk's ceil semantics produce fewer than world_size chunks for
    5/6/9 frames at world size 4, which is why this helper exists.
    """
    sizes = _sp_frame_split_sizes(num_frames, world_size)

    assert len(sizes) == world_size
    assert all(size >= 1 for size in sizes)
    assert sum(sizes) == num_frames
    assert max(sizes) - min(sizes) <= 1


def test_frame_split_sizes_known_layouts() -> None:
    assert _sp_frame_split_sizes(21, 2) == [11, 10]
    assert _sp_frame_split_sizes(11, 2) == [6, 5]
    assert _sp_frame_split_sizes(9, 4) == [3, 2, 2, 2]
    assert _sp_frame_split_sizes(5, 4) == [2, 1, 1, 1]


def test_frame_split_sizes_rejects_fewer_frames_than_ranks() -> None:
    with pytest.raises(ValueError, match="latent frame per rank"):
        _sp_frame_split_sizes(3, 4)


# ── linear attention state reduction ──


class _FakeSpAllReduce:
    """Two-pass stand-in for the SP all-reduce: pass 1 records each rank's
    partial, pass 2 replays their sum to every rank."""

    def __init__(self) -> None:
        self.partials: list[torch.Tensor] = []
        self.total: torch.Tensor | None = None

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.partials.append(tensor.clone())
        if self.total is not None:
            return self.total.clone()
        return tensor


def _make_attn(dim: int, num_heads: int, head_dim: int) -> SanaLinearAttention:
    attn = SanaLinearAttention(
        dim=dim, num_heads=num_heads, head_dim=head_dim, dropout=0.0, bias=False, qk_norm="rms_norm_across_heads"
    )
    attn.eval()
    torch.manual_seed(0)
    for _, param in sorted(attn.named_parameters()):
        param.data.normal_()
    return attn


def _attn_inputs(seq_len: int, dim: int, head_dim: int):
    """Second half of the sequence drawn from a shifted, scaled distribution so
    per-rank partial sums differ; a missing state reduction cannot cancel out."""
    torch.manual_seed(1)
    x = torch.randn(2, seq_len, dim)
    x[:, seq_len // 2 :] = x[:, seq_len // 2 :] * 3.0 + 1.0
    freqs_cos = torch.randn(1, seq_len, 1, head_dim)
    freqs_sin = torch.randn(1, seq_len, 1, head_dim)
    return x, (freqs_cos, freqs_sin)


def test_attn1_sp2_state_reduction_matches_dense(tp1_group, force_default_gemm, mocker) -> None:
    """Two token shards with a summed packed state must reproduce the dense
    full-sequence linear attention output."""
    dim, num_heads, head_dim, seq_len = 24, 4, 6, 12
    attn = _make_attn(dim, num_heads, head_dim)
    x, (freqs_cos, freqs_sin) = _attn_inputs(seq_len, dim, head_dim)
    bounds = [(0, 8), (8, 12)]

    with torch.no_grad():
        ref = attn(x, rotary_emb=(freqs_cos, freqs_sin))

    fake_group = _FakeSpAllReduce()
    mocker.patch(f"{_MODULE}.get_sequence_parallel_world_size", return_value=2)
    mocker.patch(f"{_MODULE}.get_sp_group", return_value=fake_group)

    def run_ranks() -> list[torch.Tensor]:
        outs = []
        for start, stop in bounds:
            local_rotary = (freqs_cos[:, start:stop], freqs_sin[:, start:stop])
            with torch.no_grad():
                outs.append(attn(x[:, start:stop], rotary_emb=local_rotary))
        return outs

    run_ranks()
    fake_group.total = fake_group.partials[0] + fake_group.partials[1]
    out = torch.cat(run_ranks(), dim=1)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_attn1_issues_single_packed_all_reduce(tp1_group, force_default_gemm, mocker) -> None:
    """One collective per forward, carrying scores and k_sum packed together."""
    dim, num_heads, head_dim, seq_len = 24, 4, 6, 8
    attn = _make_attn(dim, num_heads, head_dim)
    x, rotary_emb = _attn_inputs(seq_len, dim, head_dim)

    fake_group = _FakeSpAllReduce()
    mocker.patch(f"{_MODULE}.get_sequence_parallel_world_size", return_value=2)
    mocker.patch(f"{_MODULE}.get_sp_group", return_value=fake_group)

    with torch.no_grad():
        attn(x, rotary_emb=rotary_emb)

    assert len(fake_group.partials) == 1
    packed = fake_group.partials[0]
    assert packed.shape == (2, num_heads, head_dim, head_dim + 1)
    assert packed.dtype == torch.float32


def test_attn1_sp1_path_is_untouched(tp1_group, force_default_gemm, mocker) -> None:
    """At SP1 no collective may run: the dense operator sequence must not
    change shape or route through get_sp_group at all."""
    dim, num_heads, head_dim, seq_len = 24, 4, 6, 8
    attn = _make_attn(dim, num_heads, head_dim)
    x, rotary_emb = _attn_inputs(seq_len, dim, head_dim)

    mock_group = mocker.patch(f"{_MODULE}.get_sp_group")

    with torch.no_grad():
        attn(x, rotary_emb=rotary_emb)

    mock_group.assert_not_called()


# ── uneven frame gather ──


class _FakeSpAllGather:
    """Stand-in for the SP all-gather: returns precomputed equal-shape per-rank
    contributions, enforcing the equal-shard contract of the real collective."""

    def __init__(self, parts: list[torch.Tensor]) -> None:
        self.parts = parts

    def all_gather(self, tensor: torch.Tensor, dim: int = 0, separate_tensors: bool = False):
        assert separate_tensors, "frame gather must request per-rank tensors"
        assert all(part.shape == tensor.shape for part in self.parts), "all_gather requires equal shards"
        return [part.clone() for part in self.parts]


@pytest.mark.parametrize("num_frames,world_size", [(21, 2), (6, 2), (5, 4)])
def test_gather_frames_roundtrips_uneven_shards(num_frames, world_size, mocker) -> None:
    """shard -> pad -> all_gather -> per-rank narrow -> cat must reproduce the
    full sequence bitwise, with communication pads never reaching the output."""
    batch, hw, channels = 2, 3, 4
    torch.manual_seed(0)
    full = torch.randn(batch, num_frames * hw, channels)

    sizes = _sp_frame_split_sizes(num_frames, world_size)
    frame_view = full.unflatten(1, (num_frames, hw))
    locals_ = [t.flatten(1, 2) for t in frame_view.split(sizes, dim=1)]

    # Per-rank padded contributions as the real collective would deliver them;
    # NaN pads prove a leaked pad frame cannot go unnoticed.
    parts = []
    for local, size in zip(locals_, sizes):
        part = local.unflatten(1, (size, hw))
        pad = max(sizes) - size
        if pad:
            part = torch.cat([part, torch.full((batch, pad, hw, channels), torch.nan)], dim=1)
        parts.append(part)

    mocker.patch(f"{_MODULE}.get_sequence_parallel_world_size", return_value=world_size)
    mocker.patch(f"{_MODULE}.get_sp_group", return_value=_FakeSpAllGather(parts))
    for rank in range(world_size):
        mocker.patch(f"{_MODULE}.get_sequence_parallel_rank", return_value=rank)

        gathered = _sp_gather_frames(locals_[rank], sizes)

        torch.testing.assert_close(gathered, full, rtol=0, atol=0)
