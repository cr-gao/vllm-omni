# SANA-Video 2B

> Native text-to-video and image-to-video generation at 480p and 720p,
> plus a validated Diffusers-adapter compatibility path

## Summary

- Vendor: NVIDIA SANA team
- Models: `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`,
  `Efficient-Large-Model/SANA-Video_2B_720p_diffusers`
- Task: Text-to-video and image-to-video
- Mode: Offline inference and OpenAI-compatible online serving
- Maintainer: Community

## When to use this recipe

Use the native `SanaVideoPipeline` for T2V and
`SanaImageToVideoPipeline` for I2V. Both native pipelines support the 480p
and 720p checkpoints. Use `--diffusion-load-format diffusers` when you need
the black-box Diffusers compatibility baseline; adapter T2V is validated at
both resolutions, while adapter I2V is currently validated only at 480p.

The native pipeline loads the 480p checkpoint through
`DistributedAutoencoderKLWan` and the 720p checkpoint through
`DistributedAutoencoderKLLTX2Video`. These are vLLM-Omni distributed wrappers
around the corresponding Diffusers autoencoders, not independent VAE
implementations. The denoising loop also intentionally loads Diffusers'
`DPMSolverMultistepScheduler` from the checkpoint to preserve its scheduler
configuration and numerical behavior.

## References

- Upstream project: <https://github.com/NVlabs/Sana>
- Diffusers documentation: <https://huggingface.co/docs/diffusers/api/pipelines/sana_video>
- Offline T2V example:
  [`examples/offline_inference/text_to_video/text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py)
- Offline I2V example:
  [`examples/offline_inference/image_to_video/image_to_video.py`](../../examples/offline_inference/image_to_video/image_to_video.py)

## Hardware Support

## GPU

### 1x RTX 5090 32GB

The 720p checkpoint was validated with BF16 on one RTX 5090. An 81-frame,
1280×704, 50-step request took 33.56 seconds and reserved 23.58 GiB peak GPU
memory. The 480p VAE is decoded in FP32 to match the upstream recipe; a
9-frame, one-step 832×480 smoke run reserved 21.13 GiB.

Here, an 81-frame request at 16 FPS is the standard SANA-Video checkpoint
profile and produces approximately five seconds of video. It is not
minute-scale "long video" generation. The latter requires the separate
LongSANA/LongLive block-autoregressive workflow, which this pipeline does not
implement.

#### Native offline inference

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Efficient-Large-Model/SANA-Video_2B_720p_diffusers \
  --model-class-name SanaVideoPipeline \
  --prompt "A cat walking on the grass, facing the camera." \
  --negative-prompt "blurry, low quality, temporal artifacts" \
  --height 704 --width 1280 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 6 \
  --extra-body '{"motion_score": 30}' \
  --fps 16 --seed 42 --output sana_video_720p.mp4
```

For 480p, select `SANA-Video_2B_480p_diffusers` and use
`--height 480 --width 832`.

#### Native image-to-video inference

SANA checkpoints declare `SanaVideoPipeline` in `model_index.json`, so I2V
must be selected explicitly with `--model-class-name
SanaImageToVideoPipeline`.

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  --model-class-name SanaImageToVideoPipeline \
  --image input.jpg \
  --prompt "A cat turns toward the camera with smooth, natural motion." \
  --negative-prompt "blurry, low quality, temporal artifacts" \
  --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 6 \
  --extra-body '{"motion_score": 30}' \
  --fps 16 --seed 42 --output sana_video_i2v_480p.mp4
```

The same pipeline class supports the 720p checkpoint and its native LTX-2
VAE; use `--height 704 --width 1280`.

For online I2V serving:

```bash
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/image_to_video/run_server_sana_video.sh

INPUT_IMAGE=input.jpg OUTPUT_PATH=sana_video_i2v.mp4 \
  bash examples/online_serving/image_to_video/run_curl_sana_video.sh
```

#### Online serving

```bash
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/text_to_video/run_server_sana_video.sh

bash examples/online_serving/text_to_video/run_curl_sana_video.sh
```

To run the black-box compatibility backend for T2V, replace the server script
with `run_server_sana_video_diffusers.sh`. The same `/v1/videos` request
works; `num_frames` is adapted to Diffusers' `frames` argument. The script
selects `TORCH_SDPA` because SANA-Video uses an attention mask that the
AITER-backed Diffusers attention path does not accept.

The equivalent validated 480p I2V adapter command is:

```bash
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/image_to_video/run_server_sana_video_diffusers.sh

INPUT_IMAGE=input.jpg OUTPUT_PATH=sana_video_i2v_adapter.mp4 \
  bash examples/online_serving/image_to_video/run_curl_sana_video.sh
```

Do not use this adapter command as evidence of 720p I2V support: that
combination has not yet completed the end-to-end serving validation.

#### Validation boundary

The current automated serving coverage is intentionally narrower than the
pipeline registry:

| Backend | 480p T2V | 720p T2V | 480p I2V | 720p I2V |
|---|---|---|---|---|
| Native vLLM-Omni | Validated | Validated | Validated | Validated |
| Diffusers adapter | Validated | Validated | Validated | Not yet validated; not claimed |

Use the native `SanaVideoPipeline` and `SanaImageToVideoPipeline` for the
supported SANA execution paths. The Diffusers adapter is retained as a
compatibility/reference backend; the table above describes automated
validation coverage and must not be read as a broader adapter support claim.

#### Known limitations

- Sequence/tensor/CFG parallelism, Cache-DiT, TeaCache, and step execution are
  not validated for the native pipeline.
- The Diffusers backend is a compatibility path and does not provide native
  vLLM-Omni parallelism or continuous batching.
- The native pipeline still uses the checkpoint-compatible Diffusers
  scheduler and Diffusers-based VAE modules inside vLLM-Omni distributed VAE
  wrappers; "native" describes pipeline and Transformer ownership, not a
  zero-Diffusers dependency guarantee.
