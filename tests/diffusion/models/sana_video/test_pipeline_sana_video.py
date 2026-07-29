# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _DummyComponent:
    pass


def _make_request_batch(prompt, request_id="sana-video-test", **sampling_overrides):
    sampling = OmniDiffusionSamplingParams(**sampling_overrides)
    request = OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id=request_id)
    return DiffusionRequestBatch([request])


def test_sana_video_pipeline_import_and_registry():
    from vllm_omni.diffusion.models.sana_video import (
        SanaImageToVideoPipeline,
        SanaVideoPipeline,
        SanaVideoTransformer3DModel,
        get_sana_video_i2v_post_process_func,
        get_sana_video_i2v_pre_process_func,
        get_sana_video_post_process_func,
    )
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )

    assert SanaImageToVideoPipeline is not None
    assert SanaVideoPipeline is not None
    assert SanaVideoTransformer3DModel is not None
    assert get_sana_video_i2v_post_process_func is not None
    assert get_sana_video_i2v_pre_process_func is not None
    assert get_sana_video_post_process_func is not None
    assert _DIFFUSION_MODELS["SanaVideoPipeline"] == (
        "sana_video",
        "pipeline_sana_video",
        "SanaVideoPipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["SanaVideoPipeline"] == "get_sana_video_post_process_func"
    assert _DIFFUSION_MODELS["SanaImageToVideoPipeline"] == (
        "sana_video",
        "pipeline_sana_video_i2v",
        "SanaImageToVideoPipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["SanaImageToVideoPipeline"] == "get_sana_video_i2v_post_process_func"
    assert _DIFFUSION_PRE_PROCESS_FUNCS["SanaImageToVideoPipeline"] == "get_sana_video_i2v_pre_process_func"


def test_component_discovery_declarations():
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    assert SanaVideoPipeline._dit_modules == ["transformer"]
    assert SanaVideoPipeline._encoder_modules == ["text_encoder"]
    assert SanaVideoPipeline._vae_modules == ["vae"]
    assert SanaVideoPipeline.supports_step_execution is False


def test_sana_video_declares_extra_body_params():
    from vllm_omni.model_extras import get_extra_body_params

    assert get_extra_body_params("SanaVideoPipeline") == {
        "clean_caption",
        "motion_score",
        "use_resolution_binning",
    }
    assert get_extra_body_params("SanaImageToVideoPipeline") == {
        "clean_caption",
        "motion_score",
        "use_resolution_binning",
    }


def test_sana_video_i2v_preprocesses_image_and_preserves_aspect_ratio():
    from vllm_omni.diffusion.models.sana_video import get_sana_video_i2v_pre_process_func

    request = OmniDiffusionRequest(
        prompt={"prompt": "a robot walks", "multi_modal_data": {"image": Image.new("RGB", (640, 360))}},
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="sana-video-i2v-preprocess",
    )
    processed = get_sana_video_i2v_pre_process_func(SimpleNamespace())(request)

    assert processed.sampling_params.height == 448
    assert processed.sampling_params.width == 832
    assert processed.prompt["multi_modal_data"]["image"].size == (832, 448)


def test_sana_video_i2v_remote_720p_uses_loaded_transformer_config():
    from vllm_omni.diffusion.models.sana_video import get_sana_video_i2v_pre_process_func

    od_config = SimpleNamespace(
        model="Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        tf_model_config={"sample_size": 22},
    )
    request = OmniDiffusionRequest(
        prompt={"prompt": "a robot walks", "multi_modal_data": {"image": Image.new("RGB", (1280, 704))}},
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="sana-video-i2v-remote-720p",
    )

    processed = get_sana_video_i2v_pre_process_func(od_config)(request)

    assert processed.sampling_params.height == 704
    assert processed.sampling_params.width == 1280
    assert processed.prompt["multi_modal_data"]["image"].size == (1280, 704)


def test_sana_video_720p_model_id_fallback(monkeypatch):
    from vllm_omni.diffusion.models.sana_video import pipeline_sana_video

    monkeypatch.setattr(pipeline_sana_video, "get_hf_file_to_dict", lambda *_args, **_kwargs: None)
    od_config = SimpleNamespace(
        model="Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        tf_model_config={},
    )

    assert pipeline_sana_video.resolve_sana_video_sample_size(od_config) == 22


def test_sana_video_i2v_forward_maps_image_request():
    from vllm_omni.diffusion.models.sana_video import SanaImageToVideoPipeline

    pipeline = object.__new__(SanaImageToVideoPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(sample_size=30))
    calls = []

    def fake_generate_i2v(**kwargs):
        calls.append(kwargs)
        return torch.zeros(1, 3, 9, 192, 320)

    pipeline._generate_i2v = fake_generate_i2v
    image = Image.new("RGB", (320, 192))
    req = _make_request_batch(
        {"prompt": "a robot walks", "negative_prompt": "blurry", "multi_modal_data": {"image": image}},
        height=192,
        width=320,
        num_frames=9,
        num_inference_steps=2,
        guidance_scale=4.5,
        seed=42,
        extra_args={"motion_score": 30, "use_resolution_binning": False},
    )

    output = pipeline.forward(req)

    assert output.output.shape == (1, 3, 9, 192, 320)
    assert calls[0]["image"] is image
    assert calls[0]["prompt"] == "a robot walks motion score: 30."
    assert calls[0]["negative_prompt"] == "blurry"
    assert calls[0]["frames"] == 9
    assert calls[0]["generator"].initial_seed() == 42


def test_diffusers_adapter_maps_num_frames_to_frames():
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import SanaVideoPipelineUtils

    adapter = object.__new__(DiffusersAdapterPipeline)
    adapter._accept_call_kwargs = {"prompt", "negative_prompt", "frames", "generator"}
    adapter._pipeline_utils = SanaVideoPipelineUtils()
    adapter.od_config = SimpleNamespace(diffusers_call_kwargs={}, output_type=None)

    kwargs = adapter._build_call_kwargs(
        _make_request_batch(
            {"prompt": "a robot walks", "negative_prompt": "blurry"},
            num_frames=81,
            seed=42,
            generator_device="cpu",
        )
    )

    assert kwargs["frames"] == 81
    assert "num_frames" not in kwargs
    assert kwargs["generator"].initial_seed() == 42


@pytest.mark.parametrize("pipeline_class_name", ["SanaVideoPipeline", "SanaImageToVideoPipeline"])
def test_diffusers_adapter_selects_sana_pipeline_utils(pipeline_class_name):
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import (
        SanaVideoPipelineUtils,
        get_pipeline_utils,
    )

    assert isinstance(get_pipeline_utils(pipeline_class_name), SanaVideoPipelineUtils)


def test_diffusers_adapter_disables_resolution_binning_for_warmup():
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import SanaVideoPipelineUtils

    adapter = object.__new__(DiffusersAdapterPipeline)
    adapter._accept_call_kwargs = {"prompt", "use_resolution_binning", "generator"}
    adapter._pipeline_utils = SanaVideoPipelineUtils()
    adapter.od_config = SimpleNamespace(diffusers_call_kwargs={}, output_type=None)

    kwargs = adapter._build_call_kwargs(
        _make_request_batch(
            "dummy run",
            request_id=DUMMY_DIFFUSION_REQUEST_ID,
            generator_device="cpu",
        )
    )

    assert kwargs["use_resolution_binning"] is False


def test_pipeline_is_torch_module_and_supports_eval():
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    pipeline = object.__new__(SanaVideoPipeline)
    nn.Module.__init__(pipeline)

    assert isinstance(pipeline, nn.Module)
    assert pipeline.eval() is pipeline
    assert pipeline.training is False


@pytest.mark.parametrize(
    ("vae_scale", "patch_size", "valid_size", "invalid_size", "alignment"),
    [
        (8, (1, 2, 2), 624, 632, 16),
        (32, (1, 1, 1), 704, 712, 32),
    ],
)
def test_check_inputs_uses_variant_spatial_alignment(vae_scale, patch_size, valid_size, invalid_size, alignment):
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    pipeline = object.__new__(SanaVideoPipeline)
    pipeline.vae_scale_factor_spatial = vae_scale
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(patch_size=patch_size))

    pipeline.check_inputs(prompt="test", height=valid_size, width=valid_size)
    with pytest.raises(ValueError, match=f"divisible by {alignment}"):
        pipeline.check_inputs(prompt="test", height=invalid_size, width=valid_size)


def test_forward_maps_omni_request_to_sana_generation_args():
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    pipeline = object.__new__(SanaVideoPipeline)
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(sample_size=30))
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frames=torch.zeros(1, 3, 9, 192, 320))

    pipeline._generate = fake_generate
    req = _make_request_batch(
        {"prompt": "a robot walks", "negative_prompt": "blurry"},
        height=192,
        width=320,
        num_frames=9,
        num_inference_steps=2,
        guidance_scale=4.5,
        seed=42,
        extra_args={"motion_score": 30, "use_resolution_binning": False},
    )

    output = pipeline.forward(req)

    assert output.output.shape == (1, 3, 9, 192, 320)
    assert calls[0]["prompt"] == "a robot walks motion score: 30."
    assert calls[0]["negative_prompt"] == "blurry"
    assert calls[0]["height"] == 192
    assert calls[0]["width"] == 320
    assert calls[0]["frames"] == 9
    assert calls[0]["num_inference_steps"] == 2
    assert calls[0]["guidance_scale"] == 4.5
    assert calls[0]["use_resolution_binning"] is False
    assert calls[0]["generator"].initial_seed() == 42


def test_t2v_720p_uses_variant_default_resolution():
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    pipeline = object.__new__(SanaVideoPipeline)
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(sample_size=22))
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frames=torch.zeros(1, 3, 9, 704, 1280))

    pipeline._generate = fake_generate
    output = pipeline.forward(
        _make_request_batch(
            "a robot walks",
            num_frames=9,
            num_inference_steps=1,
            guidance_scale=1.0,
            seed=42,
            generator_device="cpu",
        )
    )

    assert output.output.shape == (1, 3, 9, 704, 1280)
    assert calls[0]["height"] == 704
    assert calls[0]["width"] == 1280


def test_forward_requires_exactly_one_nonempty_prompt():
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    pipeline = object.__new__(SanaVideoPipeline)

    with pytest.raises(ValueError, match="Prompt is required"):
        pipeline.forward(_make_request_batch(""))

    first = OmniDiffusionRequest(
        prompt="first",
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="first",
    )
    second = OmniDiffusionRequest(
        prompt="second",
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="second",
    )
    with pytest.raises(ValueError, match="exactly one prompt"):
        pipeline.forward(DiffusionRequestBatch([first, second]))
