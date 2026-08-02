# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from dataclasses import dataclass, field

import cache_dit
import pytest
import torch
from cache_dit import ForwardPattern
from torch import nn

import vllm_omni.diffusion.cache.cachedit.backend as cachedit_backend_module
import vllm_omni.diffusion.offloader.layerwise_backend as layerwise_backend_module
from vllm_omni.diffusion.attention import selector as attention_selector
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
from vllm_omni.diffusion.models.sana_video import pipeline_sana_video as pipeline_module
from vllm_omni.diffusion.models.sana_video.pipeline_sana_video import (
    SanaVideoPipeline,
    _validate_cache_offload_parallelism,
)
from vllm_omni.diffusion.models.sana_video.transformer_sana_video import SanaVideoTransformer3DModel
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend, LayerwiseOffloadHook
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@dataclass
class _ParallelConfig:
    tensor_parallel_size: int = 1
    cfg_parallel_size: int = 1
    sequence_parallel_size: int = 1
    data_parallel_size: int = 1


@dataclass
class _ODConfig:
    cache_backend: str = "none"
    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False
    enable_distributed_layerwise_offload: bool = False
    parallel_config: _ParallelConfig = field(default_factory=_ParallelConfig)


@dataclass
class _PipelineWithTransformer:
    transformer: nn.Module


@dataclass
class _ModelLoadConfig:
    model: str
    dtype: torch.dtype


def _tiny_transformer(monkeypatch) -> SanaVideoTransformer3DModel:
    monkeypatch.setattr(
        attention_selector,
        "_cached_get_backend_cls",
        lambda *_args, **_kwargs: SDPABackend,
    )
    return SanaVideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=4,
        cross_attention_dim=8,
        caption_channels=8,
        mlp_ratio=1.0,
        sample_size=4,
        patch_size=(1, 2, 2),
        rope_max_seq_len=16,
    )


def _parallel_config(**overrides):
    values = {
        "tensor_parallel_size": 1,
        "cfg_parallel_size": 1,
        "sequence_parallel_size": 1,
        "data_parallel_size": 1,
    }
    values.update(overrides)
    return _ParallelConfig(**values)


def _od_config(**overrides):
    values = {
        "cache_backend": "none",
        "enable_cpu_offload": False,
        "enable_layerwise_offload": False,
        "enable_distributed_layerwise_offload": False,
        "parallel_config": _parallel_config(),
    }
    values.update(overrides)
    return _ODConfig(**values)


def _record_cache_adapters(monkeypatch):
    enabled_adapters = []
    original_enable_cache = cachedit_backend_module.cache_dit.enable_cache

    def record_enable_cache(block_adapter, **kwargs):
        enabled_adapters.append(block_adapter)
        return original_enable_cache(block_adapter, **kwargs)

    monkeypatch.setattr(cachedit_backend_module.cache_dit, "enable_cache", record_enable_cache)
    return enabled_adapters


def test_sana_video_declares_cache_and_layerwise_metadata():
    adapter_config = SanaVideoTransformer3DModel._cache_dit_adapter_config

    assert adapter_config.block_forward_patterns == {
        "transformer_blocks": ForwardPattern.Pattern_3,
    }
    assert adapter_config.has_separate_cfg is False
    assert adapter_config.cached_adapter_cls is None
    assert adapter_config.check_forward_pattern is True
    assert SanaVideoTransformer3DModel._layerwise_offload_blocks_attrs == ["transformer_blocks"]
    assert SanaVideoPipeline.default_num_inference_steps == 50


def test_two_layer_tiny_transformer_enables_cache_dit(monkeypatch):
    transformer = _tiny_transformer(monkeypatch)
    pipeline = _PipelineWithTransformer(transformer=transformer)
    enabled_adapters = _record_cache_adapters(monkeypatch)
    backend = CacheDiTBackend()

    try:
        backend.enable(pipeline)

        assert backend.is_enabled()
        assert transformer._is_cached is True
        assert len(enabled_adapters) == 1
        selected_blocks = cache_dit.BlockAdapter.flatten(enabled_adapters[0].blocks)
        assert selected_blocks == [transformer.transformer_blocks]
        assert len(selected_blocks[0]) == 2
    finally:
        if enabled_adapters:
            cache_dit.disable_cache(enabled_adapters[0])


def test_cache_dit_pattern_mismatch_fails_closed(monkeypatch):
    transformer = _tiny_transformer(monkeypatch)
    transformer.transformer_blocks[1].forward = lambda unexpected_input: unexpected_input
    pipeline = _PipelineWithTransformer(transformer=transformer)
    backend = CacheDiTBackend()

    with pytest.raises(AssertionError, match="No block forward pattern matched"):
        backend.enable(pipeline)

    assert backend.is_enabled() is False
    assert not getattr(transformer, "_is_cached", False)


@pytest.mark.parametrize(
    ("feature", "parallel_field"),
    [
        ("cache", "tensor_parallel_size"),
        ("cache", "cfg_parallel_size"),
        ("cache", "sequence_parallel_size"),
        ("model_offload", "tensor_parallel_size"),
        ("model_offload", "cfg_parallel_size"),
        ("model_offload", "sequence_parallel_size"),
        ("layerwise_offload", "tensor_parallel_size"),
        ("layerwise_offload", "cfg_parallel_size"),
        ("layerwise_offload", "sequence_parallel_size"),
    ],
)
def test_cache_offload_distributed_combinations_fail_closed(feature, parallel_field):
    feature_flags = {
        "cache": {"cache_backend": "cache_dit"},
        "model_offload": {"enable_cpu_offload": True},
        "layerwise_offload": {"enable_layerwise_offload": True},
    }
    config = _od_config(
        **feature_flags[feature],
        parallel_config=_parallel_config(**{parallel_field: 2}),
    )

    with pytest.raises(NotImplementedError, match="supported only with TP1, CFG1, and SP1"):
        _validate_cache_offload_parallelism(config)


@pytest.mark.parametrize("cache_backend", [None, "", "tea_cache"])
def test_sana_video_rejects_unvalidated_cache_backends(cache_backend):
    with pytest.raises(NotImplementedError, match="Cache backend .* is not supported"):
        _validate_cache_offload_parallelism(_od_config(cache_backend=cache_backend))


def test_parallel_validation_runs_before_component_loading(monkeypatch):
    load_calls = []
    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        SanaVideoPipeline,
        "_load_components",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )
    config = _od_config(
        cache_backend="cache_dit",
        parallel_config=_parallel_config(tensor_parallel_size=2),
    )

    with pytest.raises(NotImplementedError, match="tensor_parallel_size"):
        SanaVideoPipeline(od_config=config)

    assert load_calls == []


def test_distributed_layerwise_offload_fails_before_component_loading(monkeypatch):
    load_calls = []
    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        SanaVideoPipeline,
        "_load_components",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )

    with pytest.raises(NotImplementedError, match="does not support distributed layerwise offload"):
        SanaVideoPipeline(
            od_config=_od_config(
                enable_distributed_layerwise_offload=True,
                parallel_config=_parallel_config(data_parallel_size=2),
            )
        )

    assert load_calls == []


class _TrackingModule:
    def __init__(self):
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


@pytest.mark.parametrize("component_load_device", [torch.device("cpu"), torch.device("cuda:3")])
def test_component_loading_uses_loader_device_not_runtime_device(monkeypatch, component_load_device):
    text_encoder = _TrackingModule()
    vae = _TrackingModule()
    transformer = _TrackingModule()
    tokenizer = object()
    scheduler = object()
    loaded_components = iter([text_encoder, vae])

    class _FakeVAEClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("from_pretrained_with_prefetch should invoke this loader")

    monkeypatch.setattr(torch, "get_default_device", lambda: component_load_device)
    monkeypatch.setattr(pipeline_module, "prefetch_subfolders", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_module, "_load_sana_tokenizer", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(
        pipeline_module,
        "from_pretrained_with_prefetch",
        lambda *args, **kwargs: next(loaded_components),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_load_json",
        lambda _model, filename, _local: ({"vae": [None, "FakeVAE"]} if filename == "model_index.json" else {}),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_vae_class_and_dtype",
        lambda *_args: (_FakeVAEClass, torch.float32),
    )
    monkeypatch.setattr(
        pipeline_module.SanaVideoTransformer3DModel,
        "from_config",
        classmethod(lambda _cls, _config: transformer),
    )
    monkeypatch.setattr(
        pipeline_module.DPMSolverMultistepScheduler,
        "from_pretrained",
        lambda *args, **kwargs: scheduler,
    )

    pipeline = object.__new__(SanaVideoPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cuda:7")
    pipeline.weights_sources = []
    config = _ModelLoadConfig(model="local-sana-video", dtype=torch.bfloat16)

    loaded = pipeline._load_components(config, prefix="")

    assert loaded == (tokenizer, text_encoder, vae, transformer, scheduler)
    assert pipeline.device == torch.device("cuda:7")
    assert text_encoder.to_calls == [((component_load_device,), {})]
    assert vae.to_calls == [((component_load_device,), {})]
    assert transformer.to_calls == [
        ((), {"dtype": torch.bfloat16, "device": component_load_device}),
    ]


class _DummyStream:
    def wait_stream(self, _stream) -> None:
        return None

    def wait_event(self, _event) -> None:
        return None


class _DummyEvent:
    def record(self, _stream) -> None:
        return None


@contextmanager
def _dummy_stream(_stream):
    yield None


def _patch_layerwise_platform(monkeypatch) -> None:
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "Stream", _DummyStream)
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "Event", _DummyEvent)
    monkeypatch.setattr(
        layerwise_backend_module.current_omni_platform,
        "current_stream",
        lambda: _DummyStream(),
    )
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "stream", _dummy_stream)


def _tiny_pipeline(monkeypatch) -> SanaVideoPipeline:
    pipeline = object.__new__(SanaVideoPipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = _tiny_transformer(monkeypatch)
    pipeline.text_encoder = nn.Linear(2, 2)
    pipeline.vae = nn.Linear(2, 2)
    return pipeline


def _enable_layerwise_offload(pipeline: SanaVideoPipeline) -> LayerWiseOffloadBackend:
    backend = LayerWiseOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.LAYER_WISE, pin_cpu_memory=False),
        device=torch.device("cpu"),
    )
    backend.enable(pipeline)
    return backend


def test_two_layer_offload_recovers_from_cached_block_skip_and_cleans_up(monkeypatch):
    _patch_layerwise_platform(monkeypatch)
    pipeline = _tiny_pipeline(monkeypatch)
    discovered = ModuleDiscovery.discover(pipeline)
    assert discovered.dits == [pipeline.transformer]
    assert discovered.encoders == [pipeline.text_encoder]
    assert discovered.vaes == [pipeline.vae]

    backend = _enable_layerwise_offload(pipeline)
    blocks = list(pipeline.transformer.transformer_blocks)
    assert backend.is_enabled()
    assert len(blocks) == 2

    block_1_hook = blocks[1]._hook_registry.get_hook(LayerwiseOffloadHook._HOOK_NAME)
    assert block_1_hook is not None
    assert block_1_hook.is_materialized is False

    # Cache-DiT may skip block 0, leaving block 1 without the normal prefetch.
    block_1_hook.pre_forward(blocks[1])
    assert block_1_hook.is_materialized is True
    block_1_hook.post_forward(blocks[1], None)
    assert block_1_hook.is_materialized is False

    backend.disable()
    assert backend.is_enabled() is False
    assert backend._blocks == []
    for block in blocks:
        assert block._hook_registry.get_hook(LayerwiseOffloadHook._HOOK_NAME) is None


def test_layerwise_offload_then_cache_dit_install_and_cleanup(monkeypatch):
    _patch_layerwise_platform(monkeypatch)
    pipeline = _tiny_pipeline(monkeypatch)
    offload_backend = _enable_layerwise_offload(pipeline)
    enabled_adapters = _record_cache_adapters(monkeypatch)
    cache_backend = CacheDiTBackend()
    blocks = list(pipeline.transformer.transformer_blocks)

    try:
        cache_backend.enable(pipeline)

        assert offload_backend.is_enabled()
        assert cache_backend.is_enabled()
        assert pipeline.transformer._is_cached is True
        assert enabled_adapters[0].check_forward_pattern is True
        for block in blocks:
            assert block._hook_registry.get_hook(LayerwiseOffloadHook._HOOK_NAME) is not None
    finally:
        if enabled_adapters:
            cache_dit.disable_cache(enabled_adapters[0])
        offload_backend.disable()

    assert not getattr(pipeline.transformer, "_is_cached", False)
    assert not getattr(enabled_adapters[0].pipe, "_is_cached", False)
    assert not hasattr(enabled_adapters[0].pipe, "_context_manager")
    assert offload_backend.is_enabled() is False
    for block in blocks:
        assert block._hook_registry.get_hook(LayerwiseOffloadHook._HOOK_NAME) is None
