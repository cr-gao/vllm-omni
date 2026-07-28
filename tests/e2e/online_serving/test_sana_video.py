# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SANA-Video 2B online smoke tests for the native and Diffusers backends."""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
PROMPT = "A cat walking on grass toward the camera. motion score: 30."
NEGATIVE_PROMPT = "blurry, low quality, temporal artifacts"

SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})


def _backend_cases():
    return [
        pytest.param(
            OmniServerParams(model=MODEL, server_args=["--model-class-name", "SanaVideoPipeline"]),
            id="native",
            marks=SINGLE_CARD_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--diffusion-load-format",
                    "diffusers",
                    "--diffusion-attention-backend",
                    "TORCH_SDPA",
                ],
            ),
            id="diffusers_adapter",
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
