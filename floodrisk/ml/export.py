"""Экспорт обученной U-Net в TorchScript для инференса без extras [ml].

App/inference-образ не содержит segmentation-models-pytorch (см. SRS §11.2 §4).
TorchScript-граф (`model.ts.pt`) загружается чистым torch (`torch.jit.load`),
поэтому резидентная модель работает в core-окружении. См. FR-13.
"""

from __future__ import annotations

from pathlib import Path

import torch

from floodrisk.ml import config as cfgmod


def export_unet_torchscript(cfg: dict | str, out_path: str | None = None) -> Path:
    """Грузит лучший чекпойнт U-Net и сохраняет traced TorchScript рядом (model.ts.pt).

    ``cfg`` — словарь конфига или путь к YAML.
    """
    from floodrisk.ml.model import UNetLitModule

    if isinstance(cfg, str):
        cfg = cfgmod.load_yaml(cfg)
    model = UNetLitModule(cfg)
    best_txt = Path(cfg["checkpoint"]["dirpath"]) / "best.txt"
    ckpt = best_txt.read_text(encoding="utf-8-sig").strip()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    in_ch = int(cfg["model"]["in_channels"])
    example = torch.zeros(1, in_ch, 256, 256)
    with torch.no_grad():
        traced = torch.jit.trace(model.net, example)

    out = Path(out_path) if out_path else Path(cfg["checkpoint"]["dirpath"]) / "model.ts.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    print(f"[export] TorchScript U-Net -> {out}")
    return out
