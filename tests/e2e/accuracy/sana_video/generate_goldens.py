# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate frozen SANA-Video transformer and pipeline reference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from diffusers import SanaVideoPipeline
from diffusers.utils import export_to_video
from safetensors.torch import save_file

PROMPT = "A cat walking on the grass, facing the camera. motion score: 30."
NEGATIVE_PROMPT = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, "
    "jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience."
)
VARIANTS = {
    "480p": {
        "model": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        "revision": "fed3bce411c58a0f688a31afe8f52e61acc2b15f",
        "height": 480,
        "width": 832,
        "vae_dtype": torch.float32,
        "transformer_case_shape": (1, 16, 3, 8, 8),
    },
    "720p": {
        "model": "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        "revision": "8bda5e623d0f48cd6da3b387b10ca35d15cf1c4e",
        "height": 704,
        "width": 1280,
        "vae_dtype": torch.bfloat16,
        "transformer_case_shape": (1, 128, 2, 4, 4),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def generate(variant: str, model: str | None, output_root: Path) -> Path:
    config = dict(VARIANTS[variant])
    if model:
        config["model"] = model
        config["revision"] = None
    output_dir = output_root / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = SanaVideoPipeline.from_pretrained(
        config["model"],
        revision=config["revision"],
        torch_dtype=torch.bfloat16,
    )
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(config["vae_dtype"])
    pipe.to("cuda")

    torch.manual_seed(1234)
    hidden_states = torch.randn(config["transformer_case_shape"], dtype=torch.bfloat16, device="cuda")
    encoder_hidden_states = torch.randn(1, 8, 2304, dtype=torch.bfloat16, device="cuda")
    encoder_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], device="cuda")
    timestep = torch.tensor([500.0], device="cuda")
    with torch.inference_mode():
        transformer_output = pipe.transformer(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask=encoder_attention_mask,
        ).sample

    transformer_path = output_dir / "transformer_case.safetensors"
    save_file(
        {
            "hidden_states": _cpu(hidden_states),
            "encoder_hidden_states": _cpu(encoder_hidden_states),
            "encoder_attention_mask": _cpu(encoder_attention_mask),
            "timestep": _cpu(timestep),
            "output": _cpu(transformer_output),
        },
        transformer_path,
    )

    torch.accelerator.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(42)
    torch.accelerator.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        frames = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=config["height"],
            width=config["width"],
            frames=81,
            guidance_scale=6.0,
            num_inference_steps=50,
            generator=generator,
            output_type="np",
        ).frames[0]
    torch.accelerator.synchronize()
    generation_seconds = time.perf_counter() - started

    video_path = output_dir / "pipeline.mp4"
    export_to_video(frames, video_path, fps=16)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": variant,
                "model": config["model"],
                "model_revision": config["revision"],
                "diffusers_version": __import__("diffusers").__version__,
                "prompt": PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "seed": 42,
                "height": config["height"],
                "width": config["width"],
                "num_frames": 81,
                "fps": 16,
                "num_inference_steps": 50,
                "guidance_scale": 6.0,
                "generation_seconds": generation_seconds,
                "peak_reserved_gib": torch.accelerator.max_memory_reserved() / 1024**3,
            },
            indent=2,
        )
        + "\n"
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": variant,
                "files": {
                    path.name: {"sha256": _sha256(path), "size": path.stat().st_size}
                    for path in (transformer_path, video_path, metadata_path)
                },
            },
            indent=2,
        )
        + "\n"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--model", help="Optional local checkpoint; disables the pinned Hub revision.")
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/sana-video-goldens"))
    args = parser.parse_args()
    print(generate(args.variant, args.model, args.output_root))


if __name__ == "__main__":
    main()
