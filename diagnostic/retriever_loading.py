"""Frozen evaluated dm-adapter retriever adapter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch


@dataclass
class RetrieverAdapter:
    name: str
    model: torch.nn.Module
    device: torch.device
    has_grab: bool

    def eval(self) -> None:
        self.model.eval()

    def encode_text(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(batch.to(self.device), l_aux=0)

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(batch.to(self.device), l_aux=0)

    def encode_text_grab(self, batch: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(f"Retriever {self.name} has no ablation branch")

    def encode_image_grab(self, batch: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(f"Retriever {self.name} has no ablation branch")


def load_retriever(
    retriever_name: str,
    repo_args: SimpleNamespace,
    checkpoint_path: Path,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> RetrieverAdapter:
    from model import build_model
    from utils.checkpoint import Checkpointer

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Retriever checkpoint does not exist: {checkpoint_path}")

    model = build_model(repo_args, num_classes)
    logger.info("Loading dm-adapter checkpoint from %s", checkpoint_path)
    Checkpointer(model).load(f=str(checkpoint_path))
    if device.type == "cpu":
        model = model.float()
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    logger.info("Loaded retriever=%s with normal single-branch inference", retriever_name)
    return RetrieverAdapter(name=retriever_name, model=model, device=device, has_grab=False)
