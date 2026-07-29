# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("use_local_path", [False, True])
@pytest.mark.parametrize("vae_class_name", ["AutoencoderKLWan", "AutoencoderKLLTX2Video"])
def test_sana_video_load_components_propagates_revision(monkeypatch, tmp_path, use_local_path, vae_class_name):
    from vllm_omni.diffusion.models.sana_video import pipeline_sana_video

    pinned_revision = "0123456789abcdef"
    model = str(tmp_path / "checkpoint") if use_local_path else "example/SANA-Video"
    if use_local_path:
        (tmp_path / "checkpoint").mkdir()
    expected_revision = None if use_local_path else pinned_revision

    class MovableComponent:
        def to(self, *_args, **_kwargs):
            return self

    class DummyVAE:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("from_pretrained_with_prefetch should be mocked")

    tokenizer = MovableComponent()
    text_encoder = MovableComponent()
    vae = MovableComponent()
    transformer = MovableComponent()
    scheduler = object()
    component_results = iter([tokenizer, text_encoder, vae])
    component_loads = []
    json_loads = []
    prefetch_calls = []
    resolved_vae_names = []
    scheduler_calls = []

    def fake_component_load(factory, model_arg, **kwargs):
        component_loads.append((factory, model_arg, kwargs))
        return next(component_results)

    def fake_load_json(model_arg, filename, local_files_only=True, revision=None):
        json_loads.append((model_arg, filename, local_files_only, revision))
        if filename == "model_index.json":
            return {"vae": [None, vae_class_name]}
        return {"sample_size": 30}

    monkeypatch.setattr(
        pipeline_sana_video,
        "prefetch_subfolders",
        lambda model_arg, subfolders, **kwargs: prefetch_calls.append((model_arg, tuple(subfolders), kwargs)),
    )
    monkeypatch.setattr(pipeline_sana_video, "from_pretrained_with_prefetch", fake_component_load)
    monkeypatch.setattr(pipeline_sana_video, "_load_json", fake_load_json)

    def fake_resolve_vae(class_name, _dtype):
        resolved_vae_names.append(class_name)
        return DummyVAE, torch.float32

    monkeypatch.setattr(pipeline_sana_video, "_resolve_vae_class_and_dtype", fake_resolve_vae)
    monkeypatch.setattr(
        pipeline_sana_video.SanaVideoTransformer3DModel,
        "from_config",
        lambda _config: transformer,
    )

    def fake_scheduler_load(model_arg, **kwargs):
        scheduler_calls.append((model_arg, kwargs))
        return scheduler

    monkeypatch.setattr(
        pipeline_sana_video.DPMSolverMultistepScheduler,
        "from_pretrained",
        fake_scheduler_load,
    )

    pipeline = object.__new__(pipeline_sana_video.SanaVideoPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.weights_sources = []
    result = pipeline._load_components(
        SimpleNamespace(model=model, revision=pinned_revision, dtype=torch.bfloat16),
        prefix="",
    )

    assert result == (tokenizer, text_encoder, vae, transformer, scheduler)
    assert prefetch_calls == [
        (
            model,
            ("tokenizer", "text_encoder", "vae", "scheduler"),
            {"local_files_only": use_local_path, "revision": expected_revision},
        )
    ]
    assert len(component_loads) == 3
    for _factory, model_arg, kwargs in component_loads:
        assert model_arg == model
        assert kwargs["local_files_only"] is use_local_path
        if use_local_path:
            assert "revision" not in kwargs
        else:
            assert kwargs["revision"] == pinned_revision
    assert json_loads == [
        (model, "model_index.json", use_local_path, expected_revision),
        (model, "transformer/config.json", use_local_path, expected_revision),
    ]
    assert resolved_vae_names == [vae_class_name]
    expected_scheduler_kwargs = {
        "subfolder": "scheduler",
        "local_files_only": use_local_path,
    }
    if not use_local_path:
        expected_scheduler_kwargs["revision"] = pinned_revision
    assert scheduler_calls == [(model, expected_scheduler_kwargs)]
    assert pipeline.weights_sources[0].revision == expected_revision


def test_sana_video_sample_size_uses_remote_revision(monkeypatch):
    from vllm_omni.diffusion.models.sana_video import pipeline_sana_video

    calls = []

    def fake_get_config(filename, model, **kwargs):
        calls.append((filename, model, kwargs))
        return {"sample_size": 22}

    monkeypatch.setattr(pipeline_sana_video, "get_hf_file_to_dict", fake_get_config)
    od_config = SimpleNamespace(
        model="example/SANA-Video",
        revision="0123456789abcdef",
        tf_model_config={},
    )

    assert pipeline_sana_video.resolve_sana_video_sample_size(od_config) == 22
    assert calls == [
        (
            "transformer/config.json",
            "example/SANA-Video",
            {"revision": "0123456789abcdef"},
        )
    ]
