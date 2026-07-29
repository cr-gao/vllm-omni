# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import nullcontext


def test_prefetch_subfolders_forwards_revision(monkeypatch):
    import huggingface_hub

    from vllm_omni.diffusion.model_loader import hub_prefetch

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub_prefetch, "_repo_prefetch_lock", lambda _model: nullcontext())

    hub_prefetch.prefetch_subfolders(
        "example/SANA-Video",
        ["tokenizer", "vae"],
        local_files_only=False,
        revision="0123456789abcdef",
    )

    assert calls == [
        {
            "repo_id": "example/SANA-Video",
            "allow_patterns": [
                "tokenizer/*",
                "tokenizer/**",
                "vae/*",
                "vae/**",
                "*.json",
                "*.txt",
            ],
            "revision": "0123456789abcdef",
        }
    ]


def test_from_pretrained_retry_prefetches_same_revision(monkeypatch):
    from vllm_omni.diffusion.model_loader import hub_prefetch

    attempts = 0
    prefetch_calls = []

    def flaky_factory(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("partial cache")
        return object()

    monkeypatch.setattr(hub_prefetch.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        hub_prefetch,
        "prefetch_subfolders",
        lambda model, subfolders, **kwargs: prefetch_calls.append((model, tuple(subfolders), kwargs)),
    )

    result = hub_prefetch.from_pretrained_with_prefetch(
        flaky_factory,
        "example/SANA-Video",
        subfolder="vae",
        prefetch_list=["tokenizer", "vae"],
        local_files_only=False,
        max_attempts=2,
        revision="0123456789abcdef",
    )

    assert result is not None
    assert prefetch_calls == [
        (
            "example/SANA-Video",
            ("tokenizer", "vae"),
            {
                "local_files_only": False,
                "revision": "0123456789abcdef",
            },
        )
    ]


def test_load_json_forwards_remote_revision(monkeypatch, tmp_path):
    import huggingface_hub

    from vllm_omni.diffusion.models.utils import _load_json

    cached_config = tmp_path / "config.json"
    cached_config.write_text(json.dumps({"sample_size": 30}))
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(cached_config)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    result = _load_json(
        "example/SANA-Video",
        "transformer/config.json",
        local_files_only=False,
        revision="0123456789abcdef",
    )

    assert result == {"sample_size": 30}
    assert calls == [
        {
            "repo_id": "example/SANA-Video",
            "filename": "transformer/config.json",
            "revision": "0123456789abcdef",
        }
    ]
