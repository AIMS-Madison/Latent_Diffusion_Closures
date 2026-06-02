from pathlib import Path

import torch

from project_paths import project_path, resolve_input_path, resolve_output_path
from sampling_utils import diffusion_sampler
from training_utils import create_ticks_labels, get_device, safe_cuda_synchronize


def test_project_path_resolves_relative_to_repository_root():
    expected = Path(__file__).resolve().parents[1] / "LES_NSE" / "sample.h5"
    assert project_path("LES_NSE", "sample.h5") == expected


def test_resolve_input_path_uses_env_override(monkeypatch, tmp_path):
    data_file = tmp_path / "custom.h5"
    monkeypatch.setenv("LDM_TEST_DATA", str(data_file))
    assert resolve_input_path("LDM_TEST_DATA", "LES_NSE/default.h5") == data_file


def test_resolve_output_path_creates_parent_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("LDM_OUTPUT_DIR", str(tmp_path))
    output_path = resolve_output_path("models/model.pth")
    assert output_path == tmp_path / "models" / "model.pth"
    assert output_path.parent.exists()


def test_get_device_returns_cpu_when_cuda_is_not_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device().type == "cpu"


def test_safe_cuda_synchronize_is_noop_for_cpu(monkeypatch):
    called = False

    def fake_synchronize():
        nonlocal called
        called = True

    monkeypatch.setattr(torch.cuda, "synchronize", fake_synchronize)
    safe_cuda_synchronize(torch.device("cpu"))
    assert called is False


def test_create_ticks_labels_scales_from_reference_grid():
    ticks, labels = create_ticks_labels(16, step=20, reference_size=64)
    assert ticks.tolist() == [0.0, 5.0, 10.0, 15.0]
    assert labels == ["0", "5", "10", "15"]


def test_diffusion_sampler_can_run_deterministically_without_noise():
    initial = torch.zeros(2, 4, 4)
    condition = torch.zeros_like(initial)
    time_noises = torch.tensor([1.0, 0.0])

    def marginal_prob_std(t):
        return torch.ones_like(t)

    def diffusion_coeff(t):
        return torch.ones_like(t)

    class UnitScore(torch.nn.Module):
        def forward(self, t, x, w):
            return torch.ones_like(x)

    sample = diffusion_sampler(
        condition=condition,
        score_model=UnitScore(),
        spatial_dim=4,
        marginal_prob_std=marginal_prob_std,
        diffusion_coeff=diffusion_coeff,
        batch_size=2,
        num_steps=1,
        time_noises=time_noises,
        device=torch.device("cpu"),
        initial_x=initial,
        stochastic=False,
    )

    assert torch.equal(sample, torch.ones_like(initial))
