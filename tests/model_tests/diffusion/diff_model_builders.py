from pathlib import Path

import torch
from diffusers import (
    AutoencoderKLWan,
    DPMSolverMultistepScheduler,
    SanaVideoPipeline,
    SanaVideoTransformer3DModel,
)
from transformers import Gemma2Config, Gemma2Model, GemmaTokenizerFast

from tests.helpers.tiny_model import _get_tiny_model_path, build_tiny_from_configs

TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"


def tiny_flux2_klein_builder() -> str:
    """Build a tiny Flux2Klein model from vendored configs."""
    return build_tiny_from_configs(
        "Flux2KleinPipeline", "black-forest-labs/FLUX.2-klein-4B", TINY_CONFIGS_DIR / "Flux2KleinPipeline"
    )


def tiny_ltx2_builder() -> str:
    """Build a tiny LTX2 model from vendored configs."""
    return build_tiny_from_configs("LTX2Pipeline", "Lightricks/LTX-2", TINY_CONFIGS_DIR / "LTX2Pipeline")


def tiny_sana_video_builder() -> str:
    """Build a tiny 480p SANA-Video model without downloading model weights.

    The tokenizer is the only component loaded from the upstream repository.
    All modules with weights are initialized locally from intentionally small
    configs, and the scheduler is constructed from its weight-free config.
    """
    model_id = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
    model_dir = _get_tiny_model_path("SanaVideoPipeline")

    tokenizer = GemmaTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
    scheduler = DPMSolverMultistepScheduler(
        algorithm_type="dpmsolver++",
        flow_shift=8.0,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
    )
    text_encoder = Gemma2Model(
        Gemma2Config(
            vocab_size=256000,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=512,
            sliding_window=256,
            layer_types=["sliding_attention"],
        )
    )
    transformer = SanaVideoTransformer3DModel(
        in_channels=16,
        out_channels=16,
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=1,
        num_cross_attention_heads=2,
        cross_attention_head_dim=16,
        cross_attention_dim=32,
        caption_channels=32,
        mlp_ratio=2.0,
        sample_size=30,
        patch_size=(1, 2, 2),
    )
    vae = AutoencoderKLWan(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=16,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        latents_mean=[0.0] * 16,
        latents_std=[1.0] * 16,
    )
    pipeline = SanaVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline.to(dtype=torch.bfloat16).save_pretrained(model_dir)
    return model_dir
