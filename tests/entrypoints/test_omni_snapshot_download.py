# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``omni_snapshot_download`` must honor vLLM's ``VLLM_USE_MODELSCOPE`` semantics.

vLLM treats the flag as enabled only for the literal string ``"true"``
(case-insensitive). Reading ``os.environ`` directly made every non-empty value
truthy, so an explicit opt-out such as ``VLLM_USE_MODELSCOPE=0`` still took the
ModelScope path.
"""

import sys
import types

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from vllm import envs

from vllm_omni.entrypoints import omni_base

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
PINNED_REVISION = "8bda5e623d0f48cd6da3b387b10ca35d15cf1c4e"


@pytest.fixture
def download_backend(monkeypatch: pytest.MonkeyPatch):
    """Run ``omni_snapshot_download`` and report which backend it selected.

    ModelScope is not a vLLM-Omni dependency, so the ModelScope branch is stubbed
    into ``sys.modules``; without the stub it would raise ``ModuleNotFoundError``
    instead of being observable.
    """
    # vLLM caches env lookups once a service is initialized; make sure this test
    # reads the values monkeypatch sets rather than a cached snapshot.
    envs.disable_envs_cache()

    calls: list[tuple[str, dict]] = []

    snapshot_module = types.ModuleType("modelscope.hub.snapshot_download")

    def modelscope_snapshot_download(model_id, **kwargs):
        calls.append(("modelscope", {"model_id": model_id, **kwargs}))
        return model_id

    snapshot_module.snapshot_download = modelscope_snapshot_download
    hub_module = types.ModuleType("modelscope.hub")
    hub_module.snapshot_download = snapshot_module
    root_module = types.ModuleType("modelscope")
    root_module.hub = hub_module
    for name, module in (
        ("modelscope", root_module),
        ("modelscope.hub", hub_module),
        ("modelscope.hub.snapshot_download", snapshot_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(
        omni_base,
        "download_weights_from_hf_specific",
        lambda **kwargs: calls.append(("huggingface", kwargs)),
    )

    def run(revision: str | None = None) -> str:
        calls.clear()
        omni_base.omni_snapshot_download(MODEL_ID, revision=revision)
        return calls[0][0] if calls else "none"

    return run, calls


@pytest.mark.parametrize("value", ["0", "1", "False", "false", "no", "off"])
def test_non_true_values_do_not_enable_modelscope(monkeypatch, download_backend, value):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", value)
    run, _ = download_backend

    assert envs.VLLM_USE_MODELSCOPE is False
    assert run() == "huggingface"


@pytest.mark.parametrize("value", ["true", "True", "TRUE"])
def test_true_values_enable_modelscope(monkeypatch, download_backend, value):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", value)
    run, _ = download_backend

    assert envs.VLLM_USE_MODELSCOPE is True
    assert run() == "modelscope"


def test_unset_defaults_to_huggingface(monkeypatch, download_backend):
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    run, _ = download_backend

    assert run() == "huggingface"


def test_huggingface_download_receives_pinned_revision(monkeypatch, download_backend):
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    run, calls = download_backend

    assert run(PINNED_REVISION) == "huggingface"
    assert calls == [
        (
            "huggingface",
            {
                "model_name_or_path": MODEL_ID,
                "cache_dir": None,
                "allow_patterns": ["*"],
                "revision": PINNED_REVISION,
                "require_all": True,
            },
        )
    ]


def test_huggingface_download_preserves_none_revision(monkeypatch, download_backend):
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    run, calls = download_backend

    assert run() == "huggingface"
    assert calls[0][1]["revision"] is None


def test_modelscope_download_receives_pinned_revision(monkeypatch, download_backend):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "true")
    run, calls = download_backend

    assert run(PINNED_REVISION) == "modelscope"
    assert calls == [
        (
            "modelscope",
            {
                "model_id": MODEL_ID,
                "revision": PINNED_REVISION,
            },
        )
    ]


def test_modelscope_none_revision_preserves_backend_default(monkeypatch, download_backend):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "true")
    run, calls = download_backend

    assert run() == "modelscope"
    assert calls == [("modelscope", {"model_id": MODEL_ID})]


def test_local_model_path_skips_downloads(monkeypatch, tmp_path):
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)

    def download(**_kwargs):
        pytest.fail("local model path must not be downloaded")

    monkeypatch.setattr(omni_base, "download_weights_from_hf_specific", download)

    assert omni_base.omni_snapshot_download(str(tmp_path), revision=PINNED_REVISION) == str(tmp_path)


def test_offline_commit_cache_never_falls_back_to_main(monkeypatch, tmp_path):
    """A cache with only the pinned commit must not require ``refs/main``."""
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    cached_snapshot = tmp_path / "snapshots" / PINNED_REVISION
    cached_snapshot.mkdir(parents=True)

    def download_cached_revision(*, revision, **_kwargs):
        if revision != PINNED_REVISION:
            raise LocalEntryNotFoundError("offline cache has no refs/main")
        return str(cached_snapshot)

    monkeypatch.setattr(
        omni_base,
        "download_weights_from_hf_specific",
        download_cached_revision,
    )

    assert omni_base.omni_snapshot_download(MODEL_ID, revision=PINNED_REVISION) == MODEL_ID
