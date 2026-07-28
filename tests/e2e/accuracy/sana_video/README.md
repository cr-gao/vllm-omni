# SANA-Video golden outputs

Generate each reference with Diffusers 0.38.0 and the pinned model revisions:

```bash
python tests/e2e/accuracy/sana_video/generate_goldens.py --variant 480p
python tests/e2e/accuracy/sana_video/generate_goldens.py --variant 720p
```

Upload each generated variant directory without renaming files to:

```text
s3://vllm-public-assets/omni-assets/sana-video/v1/<variant>/
```

The generated manifest contains the SHA256 and size of every artifact. Uploading
is intentionally separate from generation so CI and local test runs cannot
overwrite a frozen reference.
