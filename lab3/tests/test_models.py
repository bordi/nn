from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab3.src.config import load_config
from lab3.src.models import build_model
from lab3.src.utils import set_seed


def test_build_model_accepts_batch_sequence_feature_inputs(tmp_path: Path) -> None:
    model = build_model(load_config(project_root=tmp_path))

    inputs = torch.randn(4, 12, 1)
    outputs = model(inputs)

    assert outputs.shape == torch.Size([4])


def test_build_model_is_deterministic_with_fixed_seed(tmp_path: Path) -> None:
    config = load_config(project_root=tmp_path)

    set_seed(42)
    first_model = build_model(config)
    first_state = {name: tensor.detach().clone() for name, tensor in first_model.state_dict().items()}

    set_seed(42)
    second_model = build_model(config)

    assert set(first_state) == set(second_model.state_dict())
    for name, tensor in second_model.state_dict().items():
        assert torch.equal(first_state[name], tensor)
