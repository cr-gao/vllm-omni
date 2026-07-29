"""Lightweight contract checks for documented SANA-Video commands.

These tests only parse example documentation and scripts. They do not load a
checkpoint or claim generation-level end-to-end coverage.
"""

import ast
import re
import shlex
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
I2V_DIR = REPO_ROOT / "examples" / "offline_inference" / "image_to_video"
I2V_README = I2V_DIR / "README.md"
I2V_SCRIPT = I2V_DIR / "image_to_video.py"
T2V_DIR = REPO_ROOT / "examples" / "offline_inference" / "text_to_video"
T2V_README = T2V_DIR / "text_to_video.md"
T2V_SCRIPT = T2V_DIR / "text_to_video.py"
ONLINE_I2V_DIR = REPO_ROOT / "examples" / "online_serving" / "image_to_video"
ONLINE_I2V_README = ONLINE_I2V_DIR / "README.md"
RECIPE = REPO_ROOT / "recipes" / "NVIDIA" / "SANA-Video-2B.md"
SUPPORTED_MODELS = REPO_ROOT / "docs" / "models" / "supported_models.md"

I2V_SERVER_SCRIPT = ONLINE_I2V_DIR / "run_server_sana_video_diffusers.sh"
I2V_CURL_SCRIPT = ONLINE_I2V_DIR / "run_curl_sana_video.sh"


def _argument_value(argv: list[str], name: str) -> str:
    index = argv.index(name)
    return argv[index + 1]


def _bash_fences(markdown: str) -> list[str]:
    return re.findall(r"```bash\s*\n(.*?)```", markdown, flags=re.DOTALL)


def _sana_offline_command(markdown_path: Path, pipeline_class: str) -> list[str]:
    snippets = _bash_fences(markdown_path.read_text(encoding="utf-8"))
    commands = [
        re.sub(r"\\\s*\n", " ", snippet).strip()
        for snippet in snippets
        if "Efficient-Large-Model/SANA-Video_2B_" in snippet
        and f"--model-class-name {pipeline_class}" in snippet
        and re.search(r"(?:^|\n)python (?:examples/[^\s]+/)?(?:image|text)_to_video\.py", snippet)
    ]
    assert len(commands) == 1, f"expected one {pipeline_class} command in {markdown_path}"
    return shlex.split(commands[0])


def _parser_options(script: Path) -> set[str]:
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    return {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and argument.value.startswith("--")
    }


def _assert_documented_options_exist(argv: list[str], script: Path) -> None:
    documented_options = {argument for argument in argv if argument.startswith("--")}
    assert documented_options <= _parser_options(script)


def test_sana_video_i2v_readme_command_is_valid() -> None:
    argv = _sana_offline_command(I2V_README, "SanaImageToVideoPipeline")

    assert Path(argv[1]) == Path("image_to_video.py")
    _assert_documented_options_exist(argv, I2V_SCRIPT)
    assert _argument_value(argv, "--model-class-name") == "SanaImageToVideoPipeline"
    assert _argument_value(argv, "--height") == "480"
    assert _argument_value(argv, "--width") == "832"
    assert _argument_value(argv, "--num-frames") == "81"
    assert _argument_value(argv, "--num-inference-steps") == "50"
    assert _argument_value(argv, "--guidance-scale") == "6.0"
    assert _argument_value(argv, "--output") == "sana_video_i2v_480p.mp4"


def test_sana_video_t2v_readme_command_is_valid() -> None:
    argv = _sana_offline_command(T2V_README, "SanaVideoPipeline")

    assert Path(argv[1]) == Path("text_to_video.py")
    _assert_documented_options_exist(argv, T2V_SCRIPT)
    assert _argument_value(argv, "--model-class-name") == "SanaVideoPipeline"
    assert _argument_value(argv, "--height") == "480"
    assert _argument_value(argv, "--width") == "832"
    assert _argument_value(argv, "--num-frames") == "81"
    assert _argument_value(argv, "--num-inference-steps") == "50"
    assert _argument_value(argv, "--guidance-scale") == "6.0"
    assert _argument_value(argv, "--output") == "sana_video_480p.mp4"


def test_sana_video_recipe_offline_options_exist() -> None:
    for pipeline_class, script in (
        ("SanaVideoPipeline", T2V_SCRIPT),
        ("SanaImageToVideoPipeline", I2V_SCRIPT),
    ):
        argv = _sana_offline_command(RECIPE, pipeline_class)
        _assert_documented_options_exist(argv, script)


def test_sana_video_adapter_i2v_server_command_is_explicit() -> None:
    script = I2V_SERVER_SCRIPT.read_text(encoding="utf-8")

    assert "--model-class-name SanaImageToVideoPipeline" in script
    assert "--diffusion-load-format diffusers" in script
    assert "--diffusion-attention-backend TORCH_SDPA" in script
    assert "SANA-Video_2B_480p_diffusers" in script
    assert "SANA-Video_2B_720p_diffusers" in script
    assert "WIDTH=1280 HEIGHT=704" in script


def test_sana_video_adapter_720p_i2v_usage_is_documented() -> None:
    for markdown_path in (ONLINE_I2V_README, RECIPE):
        markdown = markdown_path.read_text(encoding="utf-8")
        assert "MODEL=Efficient-Large-Model/SANA-Video_2B_720p_diffusers" in markdown
        assert "WIDTH=1280 HEIGHT=704" in markdown


def test_sana_video_online_i2v_scripts_are_executable() -> None:
    for script_path in (I2V_SERVER_SCRIPT, I2V_CURL_SCRIPT):
        assert script_path.read_text(encoding="utf-8").startswith("#!/bin/bash\n")
        assert script_path.stat().st_mode & stat.S_IXUSR


def test_sana_video_i2v_request_uses_supported_form_fields() -> None:
    script = I2V_CURL_SCRIPT.read_text(encoding="utf-8")

    for field in (
        "input_reference=@${INPUT_IMAGE}",
        "width=${WIDTH}",
        "height=${HEIGHT}",
        "num_frames=81",
        "fps=16",
        "num_inference_steps=50",
        "guidance_scale=6.0",
        "seed=42",
        'extra_params={"motion_score":30}',
    ):
        assert f'-F "{field}"' in script or f"-F '{field}'" in script


def test_sana_video_support_matrix_matches_validation_boundary() -> None:
    recipe = RECIPE.read_text(encoding="utf-8")
    supported_models = SUPPORTED_MODELS.read_text(encoding="utf-8")

    assert "| Native vLLM-Omni | Validated | Validated | Validated | Validated |" in recipe
    assert "| Diffusers adapter | Validated | Validated | Validated | Validated |" in recipe
    assert "Diffusers adapter validated at 480p/720p" in supported_models
