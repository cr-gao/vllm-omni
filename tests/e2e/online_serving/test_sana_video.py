# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SANA-Video 2B online smoke tests for the native and Diffusers backends."""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_480P = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
MODEL_720P = "Efficient-Large-Model/SANA-Video_2B_720p_diffusers"
PROMPT = "A cat walking on grass toward the camera. motion score: 30."
NEGATIVE_PROMPT = "blurry, low quality, temporal artifacts"

SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})


def _backend_cases():
    return [
        pytest.param(
            OmniServerParams(model=MODEL_480P, server_args=["--model-class-name", "SanaVideoPipeline"]),
            id="native-480p",
            marks=SINGLE_CARD_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=MODEL_480P,
                server_args=[
                    "--diffusion-load-format",
                    "diffusers",
                    "--diffusion-attention-backend",
                    "TORCH_SDPA",
                ],
            ),
            id="diffusers-adapter-480p",
            marks=SINGLE_CARD_MARKS,
        ),
        pytest.param(
            OmniServerParams(model=MODEL_720P, server_args=["--model-class-name", "SanaVideoPipeline"]),
            id="native-720p",
            marks=SINGLE_CARD_MARKS,
        ),
    ]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _backend_cases(), indirect=True)
def test_sana_video_t2v_backends(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": 192,
            "width": 320,
            "num_frames": 9,
            "fps": 8,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "seed": 42,
        },
    }
    openai_client.send_video_diffusion_request(request_config)


def _i2v_cases():
    return [
        pytest.param(
            OmniServerParams(model=model, server_args=["--model-class-name", "SanaImageToVideoPipeline"]),
            id=variant,
            marks=SINGLE_CARD_MARKS,
        )
        for variant, model in (("480p", MODEL_480P), ("720p", MODEL_720P))
    ]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _i2v_cases(), indirect=True)
def test_sana_video_i2v_variants(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": 192,
            "width": 320,
            "num_frames": 9,
            "fps": 8,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "seed": 42,
        },
        "image_reference": (f"data:image/jpeg;base64,{generate_synthetic_image(320, 192, seed=42)['base64']}"),
    }
    openai_client.send_video_diffusion_request(request_config)
