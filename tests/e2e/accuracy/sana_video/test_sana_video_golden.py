# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import requests
import torch
from safetensors.torch import load_file
from torch import nn

from tests.e2e.accuracy.helpers import assert_video_metadata, assert_video_similarity_metrics, probe_video
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

GOLDEN_BASE_URL = os.getenv("SANA_VIDEO_GOLDEN_BASE_URL")
PROMPT = "A cat walking on the grass, facing the camera. motion score: 30."
NEGATIVE_PROMPT = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, "
    "jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience."
)
VARIANTS = {
    "480p": ("Efficient-Large-Model/SANA-Video_2B_480p_diffusers", 480, 832),
    "720p": ("Efficient-Large-Model/SANA-Video_2B_720p_diffusers", 704, 1280),
}
REVISIONS = {
    "480p": "fed3bce411c58a0f688a31afe8f52e61acc2b15f",
    "720p": "8bda5e623d0f48cd6da3b387b10ca35d15cf1c4e",
}

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(
        not GOLDEN_BASE_URL,
        reason="Set SANA_VIDEO_GOLDEN_BASE_URL after publishing the frozen v1 S3 assets.",
    ),
]


class _TransformerCheckpoint(nn.Module):
    def __init__(self, transformer: nn.Module, model: str, revision: str) -> None:
        super().__init__()
        from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

        self.transformer = transformer
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download_variant(variant: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"{GOLDEN_BASE_URL.rstrip('/')}/{variant}"
    manifest = requests.get(f"{base_url}/manifest.json", timeout=60)
    manifest.raise_for_status()
    payload = manifest.json()
    for filename, expected in payload["files"].items():
        path = output_dir / filename
        response = requests.get(f"{base_url}/{filename}", timeout=300)
        response.raise_for_status()
        path.write_bytes(response.content)
        assert path.stat().st_size == expected["size"]
        assert _sha256(path) == expected["sha256"]
    return payload


@pytest.mark.benchmark
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "_hardware",
    [pytest.param(None, marks=hardware_marks(res={"cuda": "H100"}))],
)
def test_sana_video_transformer_matches_frozen_golden(
    variant: str,
    _hardware,
    accuracy_artifact_root: Path,
) -> None:
    del _hardware
    from vllm.config import LoadConfig
    from vllm.transformers_utils.config import get_hf_file_to_dict

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel

    model, _, _ = VARIANTS[variant]
    revision = REVISIONS[variant]
    output_dir = accuracy_artifact_root / "sana_video" / variant
    _download_variant(variant, output_dir)
    case = load_file(output_dir / "transformer_case.safetensors")
    transformer_config = get_hf_file_to_dict("transformer/config.json", model, revision=revision)
    assert transformer_config is not None
    transformer = SanaVideoTransformer3DModel.from_config(transformer_config).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    checkpoint = _TransformerCheckpoint(transformer, model, revision)
    od_config = OmniDiffusionConfig(
        model=model,
        model_class_name="SanaVideoPipeline",
        dtype=torch.bfloat16,
        revision=revision,
    )
    loader = DiffusersPipelineLoader(LoadConfig(), od_config=od_config)
    transformer.load_weights(
        (name.removeprefix("transformer."), tensor)
        for name, tensor in loader.get_all_weights(checkpoint)
        if name.startswith("transformer.")
    )
    transformer.eval()
    with torch.inference_mode():
        actual = transformer(
            case["hidden_states"].to("cuda"),
            case["encoder_hidden_states"].to("cuda"),
            case["timestep"].to("cuda"),
            encoder_attention_mask=case["encoder_attention_mask"].to("cuda"),
        ).sample.float()
    expected = case["output"].to("cuda").float()
    relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
    assert relative_l2.item() <= 0.015
    assert cosine.item() >= 0.999


SERVER_CASES = [
    pytest.param(
        OmniServerParams(model=model, server_args=["--model-class-name", "SanaVideoPipeline"]),
        id=variant,
        marks=hardware_marks(res={"cuda": "H100"}),
    )
    for variant, (model, _, _) in VARIANTS.items()
]


@pytest.mark.benchmark
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_sana_video_pipeline_matches_frozen_golden(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    accuracy_artifact_root: Path,
) -> None:
    variant = "720p" if "720p" in omni_server.model else "480p"
    _, height, width = VARIANTS[variant]
    output_dir = accuracy_artifact_root / "sana_video" / variant
    _download_variant(variant, output_dir)

    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": height,
            "width": width,
            "num_frames": 81,
            "fps": 16,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "seed": 42,
        },
    }
    result = openai_client.send_video_diffusion_request(request_config)[0]
    actual_path = output_dir / "actual.mp4"
    actual_path.write_bytes(result.videos[0])
    golden_path = output_dir / "pipeline.mp4"

    metadata = probe_video(actual_path)
    assert_video_metadata(metadata, width=width, height=height, fps=16, frame_count=81)
    assert json.loads((output_dir / "metadata.json").read_text())["model"] == omni_server.model
    assert_video_similarity_metrics(
        label=f"sana_video_{variant}",
        online_path=actual_path,
        offline_path=golden_path,
        ssim_threshold=0.93,
        psnr_threshold=28.0,
    )
