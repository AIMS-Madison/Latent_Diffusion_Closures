from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange


BatchLoss = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


def fit_score_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_function: BatchLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int,
    device: torch.device,
) -> list[float]:
    """Train a conditional score model from target-condition pairs."""
    history: list[float] = []
    progress = trange(1, epochs + 1, desc="Score training")
    for epoch in progress:
        model.train()
        total_loss = 0.0
        sample_count = 0
        for target, condition in data_loader:
            target = target.to(device, non_blocking=True)
            condition = condition.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model, target, condition)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.shape[0]
            sample_count += target.shape[0]

        scheduler.step()
        average_loss = total_loss / sample_count
        history.append(average_loss)
        progress.set_description(
            f"Epoch {epoch}/{epochs} | score loss {average_loss:.5e}"
        )
    return history
