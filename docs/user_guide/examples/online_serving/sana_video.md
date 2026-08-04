# SANA-Video 2B Online Serving

vLLM-Omni supports SANA-Video 2B text-to-video (T2V) and
image-to-video (I2V) generation through native pipelines and a Diffusers
compatibility backend.

| Checkpoint | Default output | VAE |
| :--------- | :------------- | :-- |
| `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` | 832 x 480 | Wan VAE |
| `Efficient-Large-Model/SANA-Video_2B_720p_diffusers` | 1280 x 704 | LTX-2 VAE |

Both checkpoints use 81 frames at 16 FPS as their standard generation
profile, producing approximately five seconds of video. The examples below
use BF16 and a single GPU.

## Text-to-video

Start the native T2V pipeline:

```bash
vllm serve Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  --omni \
  --model-class-name SanaVideoPipeline \
  --dtype bfloat16 \
  --port 8091
```

Send a synchronous request and save the MP4 response:

```bash
curl -sS -X POST http://localhost:8091/v1/videos/sync \
  -F "prompt=A cinematic tracking shot of a sailboat crossing the ocean at sunset" \
  -F "negative_prompt=blurry, low quality, temporal artifacts" \
  -F "height=480" \
  -F "width=832" \
  -F "num_frames=81" \
  -F "fps=16" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=6.0" \
  -F "seed=42" \
  -F 'extra_params={"motion_score":30}' \
  --output sana_video.mp4
```

The same workflow is available as scripts:

```bash
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/text_to_video/run_server_sana_video.sh

bash examples/online_serving/text_to_video/run_curl_sana_video.sh
```

For 720p, select the 720p checkpoint and request `width=1280` and
`height=704`. If width and height are omitted, the native pipeline derives
the defaults from the loaded checkpoint.

## Image-to-video

The released model indexes identify the T2V class, so select the native I2V
pipeline explicitly:

```bash
vllm serve Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  --omni \
  --model-class-name SanaImageToVideoPipeline \
  --dtype bfloat16 \
  --port 8099
```

Submit an asynchronous request with an input image, poll it, and download the
result with the provided client script:

```bash
INPUT_IMAGE=/path/to/input.png \
  OUTPUT_PATH=sana_video_i2v.mp4 \
  bash examples/online_serving/image_to_video/run_curl_sana_video.sh
```

The equivalent server helper is:

```bash
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/image_to_video/run_server_sana_video.sh
```

For the 720p checkpoint, pass the corresponding dimensions to the client:

```bash
MODEL=Efficient-Large-Model/SANA-Video_2B_720p_diffusers \
  bash examples/online_serving/image_to_video/run_server_sana_video.sh

INPUT_IMAGE=/path/to/input.png WIDTH=1280 HEIGHT=704 \
  OUTPUT_PATH=sana_video_i2v_720p.mp4 \
  bash examples/online_serving/image_to_video/run_curl_sana_video.sh
```

## Diffusers compatibility backend

Use the Diffusers adapter when you need a black-box compatibility baseline.
The adapter is validated for T2V and I2V with both checkpoint variants.

T2V:

```bash
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/text_to_video/run_server_sana_video_diffusers.sh
```

I2V:

```bash
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/image_to_video/run_server_sana_video_diffusers.sh
```

These helpers select `TORCH_SDPA` because the SANA-Video attention mask is
not accepted by the AITER-backed Diffusers attention path. Requests use the
same `/v1/videos` API as the native pipelines.

## Generation parameters

| Parameter | Standard value | Description |
| :-------- | :------------- | :---------- |
| `width`, `height` | 832 x 480 or 1280 x 704 | Output dimensions for the selected checkpoint |
| `num_frames` | 81 | Number of generated frames |
| `fps` | 16 | MP4 playback frame rate |
| `num_inference_steps` | 50 | Denoising steps |
| `guidance_scale` | 6.0 | Classifier-free guidance scale |
| `seed` | 42 | Seed used by the request generator |
| `negative_prompt` | optional | Content and quality attributes to avoid |
| `extra_params.motion_score` | 30 | Motion-strength instruction appended to the prompt |

## Hardware and limitations

The native 720p, 81-frame, 50-step path was validated on one 32 GiB RTX
5090. Its observed peak reserved memory was approximately 23.6 GiB; the
corresponding Diffusers-adapter I2V request peaked at approximately 25.6 GiB.
Available memory depends on the task, backend, allocator state, and software
versions.

- Native sequence/tensor/CFG parallelism and Cache-DiT are not validated in
  this initial single-GPU implementation.
- The Diffusers adapter is a compatibility path and does not provide native
  vLLM-Omni parallelism or continuous batching.
- The native pipelines retain the checkpoint-compatible Diffusers scheduler.
  Their VAE components are loaded through vLLM-Omni distributed VAE wrappers.
- Standard 81-frame inference is not minute-scale long-video generation.

For offline commands, architecture details, measured results, and the full
support matrix, see the [SANA-Video 2B recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/NVIDIA/SANA-Video-2B.md),
the [T2V guide](../offline_inference/text_to_video.md), and the
[I2V guide](../offline_inference/image_to_video.md).
