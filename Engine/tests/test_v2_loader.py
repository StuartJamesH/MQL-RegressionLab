"""
Engine/tests/test_v2_loader.py — Tests for ModelPackLoader.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from Engine.v2.model_pack import ModelPackLoader


def _find_example_pack() -> Path:
    """Return a populated v2 transformer pack if one exists."""
    base = Path("ModelWorkbench/ModelPacks/transformers")
    if not base.exists():
        pytest.skip("No transformer model packs found")
    candidates = sorted(
        [d for d in base.iterdir() if d.is_dir() and (d / "model.onnx").exists()],
        key=lambda p: p.name,
    )
    for cand in candidates:
        if all((cand / f).exists() for f in ModelPackLoader.REQUIRED_FILES):
            return cand
    pytest.skip("No complete v2 model pack found")


@pytest.fixture(scope="module")
def example_pack_dir() -> Path:
    return _find_example_pack()


def test_load_returns_required_keys(example_pack_dir: Path) -> None:
    pack = ModelPackLoader.load(str(example_pack_dir))
    required_keys = {
        "config",
        "normalizer",
        "feature_spec",
        "model_info",
        "pytorch_state_dict",
        "onnx_path",
        "pack_dir",
    }
    assert required_keys.issubset(set(pack.keys()))


def test_config_matches_config_json(example_pack_dir: Path) -> None:
    pack = ModelPackLoader.load(str(example_pack_dir))
    with open(example_pack_dir / "config.json", "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    assert pack["config"].max_seq_len == raw_config["max_seq_len"]
    assert pack["config"].n_horizons == raw_config["n_horizons"]
    assert pack["config"].in_channels == raw_config["in_channels"]


def test_missing_onnx_raises(example_pack_dir: Path, tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete_pack"
    incomplete.mkdir()
    for fname in ModelPackLoader.REQUIRED_FILES:
        src = example_pack_dir / fname
        if src.exists():
            (incomplete / fname).write_text(src.read_text())
    with pytest.raises(FileNotFoundError):
        ModelPackLoader.load(str(incomplete))
