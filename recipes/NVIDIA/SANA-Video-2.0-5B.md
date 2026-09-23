# SANA-Video 2.0 5B

`SanaVideo2Pipeline` implements native text-to-video (T2V) and
text-image-to-video (TI2V) inference for the official **50-step**
`Efficient-Large-Model/SANA-Video_2.0_5B_720p` checkpoint. Providing one image
selects TI2V. The 4-step preview and 14B models are separate variants and are
not supported by this pipeline.

The transformer and denoising loops run in vLLM-Omni. The pipeline reuses Gemma
for text conditioning and the Diffusers LTX 2.3 VAE. T2V uses second-order
multistep flow DPM-Solver++; TI2V uses FlowMatch Euler with a clean, fixed first
latent frame and frame-dependent timesteps.

**Sequence parallelism is experimental:** real-checkpoint SP2 BF16 failed
numerical acceptance. See the precision results before using the SP example.

## Components

Default Hub revisions are pinned:

| Component | Repository | Revision |
| --- | --- | --- |
| Transformer | `Efficient-Large-Model/SANA-Video_2.0_5B_720p` | `f2d95fa06400f186fd1b077d12c69c8ac58aba48` |
| VAE | `Efficient-Large-Model/LTX-2.3-Diffusers` | `362acdf779d42e785fb26910c32254e9458e78c2` |
| Text encoder/tokenizer | `Efficient-Large-Model/gemma-2-2b-it` | `569d9809d0c8b6722d4d31b5a77a2ec7a400650a` |

A local transformer directory must contain `config.yaml` with
`model.model: SanaVideo2_5B` and
`checkpoints/SANA_Video_2.0_5B_720p.pth`. It is detected without a Diffusers
`model_index.json`. The VAE path must contain its `vae/` subdirectory. Text
encoder and tokenizer overrides point directly to their component directories.

## Native offline inference

```python
from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    engine = Omni(
        model="Efficient-Large-Model/SANA-Video_2.0_5B_720p",
        dtype="bfloat16",
        enforce_eager=True,
        # Optional local component overrides:
        # model_config={
        #     "vae_model": "/path/to/LTX-2.3-Diffusers",
        #     "text_encoder_model": "/path/to/gemma/text_encoder",
        #     "tokenizer_model": "/path/to/gemma/tokenizer",
        # },
    )
    try:
        prompt = {"prompt": "A small red boat sailing across a calm lake."}
        # For TI2V, supply exactly one RGB image:
        # prompt["multi_modal_data"] = {"image": Image.open("reference.png").convert("RGB")}
        outputs = engine.generate(
            prompt,
            OmniDiffusionSamplingParams(
                height=64,
                width=96,
                num_frames=9,
                num_inference_steps=50,
                guidance_scale=8.0,
                seed=42,
                extra_args={"motion_score": 10, "flow_shift": 12.0},
            ),
        )
        return outputs
    finally:
        engine.close()


if __name__ == "__main__":
    main()
```

The dimensions above are a small correctness smoke test, not a video quality
preset. The upstream release defaults are height 736, width 1280, 193 frames,
50 steps, CFG 8, flow shift 12, and 24 FPS. Heights and widths must be positive
multiples of 32; frame counts must satisfy `(num_frames - 1) % 8 == 0`.
The release envelope is at most 193 frames, area `736 * 1280`, and 1280 pixels
on either axis. Successful small-shape validation alone does not establish
quality or memory requirements at the release dimensions.

The pipeline adds the release prompt instruction and motion-score suffix.
When the negative prompt is omitted, it uses the release negative prompt;
passing an empty string explicitly requests empty negative conditioning.
CFG is enabled when guidance exceeds 1. Images undergo RGB conversion,
bicubic resize to fill, and center cropping inside the pipeline.

Component overrides use `model_config`, with optional `vae_revision`,
`text_encoder_revision`, and `tokenizer_revision`. `custom_pipeline_args` is a
separate framework mechanism for replacing the pipeline class.

## Current execution scope

The native default uses one NVIDIA GPU with BF16 or FP32 transformer weights.
An experimental pure-Ulysses SP path is available as described below. TP, CFG
parallelism, pipeline parallelism, distributed VAE, HSDP, quantization,
caching, and CPU/layerwise offload are rejected. Custom sigma/timestep schedules
and multiple videos per request are not implemented. Reducing the inference
step count does not turn the formal checkpoint into the 4-step preview.

## Experimental sequence parallelism

Use vLLM 0.29.0, PyTorch 2.13 and Diffusers 0.40. Keep TP=1 and CFG
parallel=1. Native batch CFG remains enabled. The transformer
splits flattened video tokens before allocating Attention Residual buffers.
Linear layers sum their FP32 states over the full SP group; softmax anchors
use the shared Ulysses communication with native FP32 SDPA. Each denoising
prediction is gathered back to all ranks. Text conditioning, patch embedding,
the sampler and VAE remain replicated.

**SP2 strict BF16 numerical acceptance failed. Do not deploy the BF16 SP
example as a validated configuration.** On the real 5B checkpoint, SP1
repeated bitwise, but SP2 final latents differed from SP1 by 0.0531–0.0577
relative L2 after 50 steps; 85 of 153 per-step latent and decoded comparisons
exceeded the pre-established SP1 BF16-versus-FP32 envelope. The examples below
exercise the experimental path; BF16 SP is not yet a validated configuration.

| Configuration | Input contract | Validation status |
| --- | --- | --- |
| SP1 | Native path | Native evidence below; small real-checkpoint BF16 repeats bitwise |
| SP2, `strict` | Token count divisible by 2 | Narrow-model NCCL FP32 and small real-checkpoint FP32 latents checked; real-checkpoint BF16 failed acceptance |
| SP2, `advanced_uaa` | At least one video token per rank | Narrow-model NCCL FP32 checked; real-checkpoint acceptance pending |
| SP4/8, `advanced_uaa` | At least one video token per rank | Experimental; GPU acceptance pending |

The released model has 10 softmax heads. SP4/SP8 require `advanced_uaa`, which
pads heads inside the shared communication strategy. Video tokens are split
unevenly without introducing attention padding. Both RoPE and TI2V frame
conditioning retain their global coordinates. Inputs with fewer tokens than
SP ranks, or uneven token counts in `strict` mode, are rejected before model
collectives. There is no Ring or AllGather-KV integration in this path.

For offline inference, add these arguments to the `Omni` constructor above:

```python
ulysses_degree=2,
ulysses_mode="advanced_uaa",
```

For an experimental two-GPU server:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Efficient-Large-Model/SANA-Video_2.0_5B_720p \
  --omni --dtype bfloat16 --enforce-eager --usp 2 --ulysses-mode advanced_uaa \
  --host 127.0.0.1 --port 8091
```

The normal startup warmup uses 512x512 and 9 frames (512 latent tokens).
The 64x96, 9-frame example has 12 latent tokens; use `advanced_uaa` for SP8.
Run SP1 and SP requests with identical model/component revisions, prompt,
image, seed, precision, schedule and guidance. Startup, back-to-back T2V/TI2V
requests and changing dimensions must all be exercised before deployment.
With vLLM 0.29.0, both SP1 and SP2 completed standard Omni startup, default
warmup and three consecutive two-step T2V/TI2V requests, including a changed
shape. This is an entrypoint smoke test, not 50-step numerical acceptance.

### Numerical acceptance and performance

The focused tests exercise actual multi-process collectives on a narrow
32-layer transformer with 24 linear layers, eight 10-head softmax anchors,
nonzero learned Attention Residual projections and batch CFG. This preserves
depth and communication structure, but does not replace real 5B checkpoint
or full-size video validation.

CPU FP32 Gloo observations (seed 8006, hidden size 120, 32 layers; all ranks):

| Check | SP2 strict max abs / relative L2 | SP4 advanced max abs / relative L2 |
| --- | --- | --- |
| T2V/TI2V transformer, consecutive shapes | `9.54e-7 / 6.94e-7` | `1.28e-6 / 5.51e-7` |
| Five-step T2V, CFG 8 | `2.62e-6 / 1.20e-6` | `2.38e-6 / 1.01e-6` |
| Five-step TI2V, CFG 8 | `2.73e-6 / 1.65e-6` | `3.10e-6 / 1.73e-6` |

These are maxima across the recorded cases/steps, compared with the same
weights on SP1. SP4 includes 15 tokens split 4/4/4/3 and head padding 10 to 12.
Neither the random narrow model nor CPU Gloo establishes GPU or video quality.

Two A800-SXM4-80GB GPUs with NVLink pass the narrow-model NCCL checks
in both SP2 modes (PyTorch 2.13, vLLM 0.29.0, Diffusers 0.40, seed 8006):

| Check | SP2 strict max abs / relative L2 | SP2 advanced max abs / relative L2 |
| --- | --- | --- |
| T2V/TI2V transformer, consecutive shapes | `6.56e-7 / 3.48e-7` | `7.75e-7 / 4.76e-7` |
| Five-step T2V, CFG 8 | `1.91e-6 / 8.10e-7` | `1.91e-6 / 8.10e-7` |
| Five-step TI2V, CFG 8 | `4.14e-6 / 2.03e-6` | `4.14e-6 / 2.03e-6` |

The matched vLLM 0.29.0 rerun reproduced these NCCL results; the Gloo SP2/SP4
CPU cases also passed. These FP32 tests disable TF32 for the patch Conv3d and
require IEEE FP32 matmul, retaining native SDPA selection and the original
`1e-5` gates.
With the default cuDNN TF32 setting, the narrow-model TI2V trajectory reached
`1.97e-5` maximum absolute error and failed the elementwise gate. Fixed-input
replay and a Conv3d precision comparison localized the amplification to TF32
rounding of the evolving latent. The production precision settings are
unchanged; the strict FP32 result does not establish default-TF32 or BF16
trajectory agreement. SP2 advanced includes 15 tokens split 8/7. SP4/SP8
NCCL remain untested on this two-GPU host.

```bash
python -m pytest -o addopts= -q tests/diffusion/models/sana_video2
python -m pytest -o addopts= -s -q \
  tests/diffusion/distributed/test_sana_video2_sp_numeric.py -k gloo
# Run only after allocating and authorizing the required NVIDIA GPUs:
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -o addopts= -s -q \
  tests/diffusion/distributed/test_sana_video2_sp_numeric.py -k nccl
```

The NCCL cases skip degrees larger than the visible device count. Repeat with
four/eight visible GPUs to exercise those degrees. Tests print maximum absolute
and relative L2 error by rank and compare every recorded sampler step. FP32
thresholds are fixed in the test; do not relax them to accommodate a failed run.

In the direct pipeline, real 5B weights were compared at three small shapes
for 50 steps with seed 42, CFG 8 and flow shift 12. Across 150 FP32 latent
pairs, worst relative L2 was `8.549824e-6` and maximum absolute error was
`1.23977e-4`. The elementwise FP32 gate did not pass at every element.
Decoding used the BF16 VAE and reached `0.0086742266` relative L2, so strict
FP32 decoded agreement is not established.
BF16 SP2 strict failed the pre-established SP1 BF16-versus-FP32 envelope as
noted above. With identical first-layer attention inputs, the first difference
appeared in BF16 QKV. Single-GPU contiguous-split replay reproduced all 15
saved stages of the actual SP2 first layer bitwise. This identifies an initial source of drift, but does
not explain or accept every later trajectory difference.

Full-size BF16 SP1/SP2 T2V and TI2V runs at 736x1280/193 frames completed
50 steps and decoding. Final latent relative L2 was `0.08282` (T2V) and
`0.16842` (TI2V); decoded relative L2 was `0.12059` and `0.20095`, respectively.
All saved tensors were finite and both SP2 ranks agreed, but these results do
not establish numerical acceptance. Full-size FP32 comparison is pending.
Performance has not been measured while numerical acceptance remains open.

For NVLink measurements, disable compile/cache/offload, use the same FP32
self-attention kernel and work on all configurations, and record the topology.
Exclude startup and warmup, repeat at least five requests, synchronize devices
at timing boundaries, and report DiT, denoising and end-to-end median/range plus
each rank's peak allocated/reserved memory. Profile computation and collectives
before selecting a recommended degree. No speedup is claimed here.

Upstream correctness reference:
`NVlabs/Sana@e93c883e10730ee5a4a6edf1cbcf501dc4ef753b`.

## Online requests

The official model ID and local release directories resolve to the same native
pipeline. Start a single-GPU server with:

```bash
vllm serve Efficient-Large-Model/SANA-Video_2.0_5B_720p \
  --omni --dtype bfloat16 --enforce-eager --host 127.0.0.1 --port 8091
```

For local component overrides, use the generic diffusion stage override:

```bash
--stage-overrides '{"0":{"model_config":{"vae_model":"/path/to/LTX-2.3-Diffusers","text_encoder_model":"/path/to/gemma/text_encoder","tokenizer_model":"/path/to/gemma/tokenizer"}}}'
```

Create a small T2V request:

```bash
curl --fail-with-body http://127.0.0.1:8091/v1/videos \
  -F 'prompt=A small red boat sailing across a calm lake.' \
  -F height=64 -F width=96 -F num_frames=9 -F fps=24 \
  -F num_inference_steps=50 -F guidance_scale=8 -F seed=42 \
  -F 'extra_params={"motion_score":10,"flow_shift":12}'
```

For TI2V, add `-F 'input_reference=@reference.png'` to the same request. Use
the returned `id` to query `/v1/videos/{id}` and, once completed, download
`/v1/videos/{id}/content`. Omitting `num_frames` selects 193; this default does
not prohibit shorter legal clips. The service preserves the source image's
geometry for the pipeline's own resize and center crop.

## Validation scope

The following results are recorded native SP1 baseline evidence from the
original integration, not new measurements of the SP path. They used one
NVIDIA H20-3e with BF16 weights and the release FP32 self-attention path:

| Check | Scope | Result |
| --- | --- | --- |
| Upstream sampler comparison | T2V/TI2V, steps 2/5/50, shifts 1/12 | Exact trajectories |
| Real-weight upstream comparison | 64x96, 9 frames, 50 steps, CFG 8 | Exact step latents and decoded outputs |
| Native offline and HTTP video API | T2V/TI2V, 64x96, 9 frames | Generation and MP4 export/download pass |
| Full release dimensions | T2V/TI2V, 736x1280, 193 frames, 50 steps, CFG 8 | Finite decoded outputs; 193-frame MP4 files at 24 FPS |

The full-dimension runs used approximately 32.42 GiB peak allocated GPU
memory. Observed wall times were 498 seconds for T2V and 496 seconds for TI2V,
including text/image conditioning, denoising, VAE decoding and saving the raw
output tensor, but excluding MP4 encoding. These are single-run capacity and
correctness observations, not a repeated latency benchmark or a speedup claim.
Full-dimension output was not compared numerically with an upstream full video;
the exact real-weight comparison above used the stated small dimensions.
