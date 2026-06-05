"""OQ-8 / NFR-7: дисперсия метрик U-Net между случайными сидами.

Обучает U-Net несколько раз с разными ``training.seed`` (одинаковый рецепт и данные),
меряет разброс IoU/F1 на test-split и пишет `reports/seed_variance.md`. Отвечает на
NFR-7 («воспроизводимость метрик ±2% IoU»), помеченный в acceptance как «не измерялась».

ВАЖНО — продакшн-инвариант: каждый сид пишет ckpt/best.txt/TorchScript в ОТДЕЛЬНЫЙ
`models/unet/oq8/seed<N>/` (см. train_unet/export_unet_torchscript), поэтому
продакшн-модель `models/unet/v1/` НЕ затрагивается. norm_stats берётся общий (v1) —
данные те же, пересчёт не нужен.

Схема укорочена (max_epochs=12 вместо 20) ради счёта на CPU: плато val_iou достигается
уже к ~7 эпохе (продакшн-чекпойнт — epoch=07). Меряем чувствительность РЕЦЕПТА к сиду,
не абсолютный максимум.

Запуск (из корня, окружение [ml]):
    python scripts/oq8_seed_variance.py
Опц.: SEEDS=42,1,2 MAX_EPOCHS=12 python scripts/oq8_seed_variance.py
"""

from __future__ import annotations

import copy
import os
import statistics
from pathlib import Path

import numpy as np

from floodrisk.ml import config as cfgmod
from floodrisk.ml import metrics as M
from floodrisk.ml.evaluate import _best_f1_threshold, _unet_predictions
from floodrisk.ml.train import _ensure_norm_stats, train_unet

BASE_CONFIG = "configs/unet_v1.yaml"
REPORT = Path("reports/seed_variance.md")


def _seed_config(base: dict, seed: int, max_epochs: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["training"]["seed"] = seed
    cfg["training"]["max_epochs"] = max_epochs
    cfg["checkpoint"]["dirpath"] = f"models/unet/oq8/seed{seed}/"
    cfg["mlflow"]["experiment_name"] = "floodrisk_oq8"
    return cfg


def _evaluate(cfg: dict) -> dict:
    """Метрики одного обученного сида: IoU/F1@0.5, PR-AUC, и F1 на best-F1-пороге (по val)."""
    ut, up = _unet_predictions(cfg, "test")
    yt, pp = np.concatenate(ut), np.concatenate(up)
    point = M.compute_all(yt, pp)

    uv_t, uv_p = _unet_predictions(cfg, "val")
    thr = _best_f1_threshold(uv_t, uv_p)
    return {
        "iou": point["IoU"],
        "f1": point["F1"],
        "pr_auc": point["PR-AUC"],
        "thr_star": thr,
        "iou_star": M.iou_score(yt, pp, thr),
        "f1_star": M.f1_score_bin(yt, pp, thr),
    }


def _spread(vals: list[float]) -> tuple[float, float, float, float]:
    """(mean, std, min, max). std — выборочное (n-1); при n<2 → 0."""
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std, min(vals), max(vals)


def _write_report(rows: list[dict], max_epochs: int) -> None:
    lines: list[str] = []
    lines.append("# OQ-8 / NFR-7 — дисперсия метрик U-Net между сидами\n")
    lines.append(
        f"U-Net обучен {len(rows)} раз с разными `training.seed` на одном датасете "
        f"(Тулун-2019, v1) и одинаковом рецепте (`max_epochs={max_epochs}`, BCE+Dice, "
        "pos_weight=20). Метрики — попиксельно по пулу test (32 тайла), порог 0.5; "
        "`*` — на пороге максимума F1 по val. Цель — оценить чувствительность к сиду "
        "(инициализация + порядок аугментаций), NFR-7 (±2% IoU).\n"
    )
    lines.append("\n## Метрики по сидам\n")
    lines.append("| seed | IoU@0.5 | F1@0.5 | PR-AUC | порог* | IoU@порог* | F1@порог* |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['iou']:.4f} | {r['f1']:.4f} | {r['pr_auc']:.4f} | "
            f"{r['thr_star']:.2f} | {r['iou_star']:.4f} | {r['f1_star']:.4f} |"
        )

    iou_mean, iou_std, iou_lo, iou_hi = _spread([r["iou"] for r in rows])
    f1_mean, f1_std, f1_lo, f1_hi = _spread([r["f1"] for r in rows])
    lines.append("\n## Разброс (IoU/F1 @0.5)\n")
    lines.append("| метрика | mean | std | min | max | размах |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| IoU | {iou_mean:.4f} | {iou_std:.4f} | {iou_lo:.4f} | {iou_hi:.4f} | "
        f"{iou_hi - iou_lo:.4f} |"
    )
    lines.append(
        f"| F1 | {f1_mean:.4f} | {f1_std:.4f} | {f1_lo:.4f} | {f1_hi:.4f} | {f1_hi - f1_lo:.4f} |"
    )

    within = (iou_hi - iou_lo) <= 0.02
    verdict = (
        "✅ размах IoU ≤ 0.02 — укладывается в порог NFR-7 (±2%)."
        if within
        else "⚠️ размах IoU > 0.02 — превышает порог NFR-7 (±2%); honest-результат: "
        "на одном маленьком событии (32 test-тайла, flood ~1.2%) метрики чувствительны к сиду. "
        "Это согласуется с выводом M-2 (узкое место — данные): больше событий/регионов → "
        "стабильнее метрики."
    )
    lines.append(f"\n## Вывод (NFR-7)\n\n{verdict}\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[oq8] отчёт → {REPORT}")


def main() -> int:
    seeds = [int(s) for s in os.environ.get("SEEDS", "42,1,2").split(",")]
    max_epochs = int(os.environ.get("MAX_EPOCHS", "12"))
    base = cfgmod.load_yaml(BASE_CONFIG)

    rows: list[dict] = []
    for seed in seeds:
        print(f"\n===== OQ-8 seed={seed} (max_epochs={max_epochs}) =====")
        cfg = _seed_config(base, seed, max_epochs)
        _ensure_norm_stats(cfg)  # общий v1-файл → присутствует → пропуск
        train_unet(cfg)
        metrics = _evaluate(cfg)
        rows.append({"seed": seed, **metrics})
        print(f"[oq8] seed={seed}: IoU={metrics['iou']:.4f} F1={metrics['f1']:.4f}")
        _write_report(rows, max_epochs)  # инкрементально: частичный результат переживает обрыв

    print("\n[oq8] готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
