"""Сравнение U-Net vs baseline на test-split → reports/comparison_v1.md.

См. SRS §7.5, FR-7, OQ-6 (bootstrap-CI 95%), M-2 (§15).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from floodrisk.ml import config as cfgmod
from floodrisk.ml import metrics as M
from floodrisk.ml.data import _read_label, _read_tile, load_norm_stats, tile_paths

METRIC_ORDER = ["IoU", "F1", "ROC-AUC", "PR-AUC", "Brier"]
# M-2: U-Net должен превзойти baseline по этим метрикам (больше = лучше) с непересекающимися CI.
M2_METRICS = ["IoU", "F1", "PR-AUC"]


def _unet_predictions(cfg: dict, split: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    import torch

    from floodrisk.ml.model import UNetLitModule

    best_txt = Path(cfg["checkpoint"]["dirpath"]) / "best.txt"
    ckpt_path = best_txt.read_text(encoding="utf-8").strip()
    model = UNetLitModule(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    mean, std = load_norm_stats(cfg)
    trues: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for _tid, tpath, lpath in tile_paths(cfg, split):
            x = ((_read_tile(tpath) - mean) / std)[None]  # [1,7,H,W]
            logit = model(torch.from_numpy(x.astype("float32")))
            prob = torch.sigmoid(logit)[0, 0].numpy().reshape(-1)
            trues.append(_read_label(lpath).reshape(-1))
            probs.append(prob)
    return trues, probs


def _baseline_predictions(cfg: dict, split: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    from floodrisk.ml.baseline import load_baseline

    clf = load_baseline(cfg["artifact_path"])
    mean, std = load_norm_stats(cfg)
    trues: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for _tid, tpath, lpath in tile_paths(cfg, split):
        x = ((_read_tile(tpath) - mean) / std).reshape(7, -1).T  # [HW,7]
        prob = clf.predict_proba(x)[:, 1]
        trues.append(_read_label(lpath).reshape(-1))
        probs.append(prob)
    return trues, probs


def _best_f1_threshold(trues: list[np.ndarray], probs: list[np.ndarray]) -> float:
    """Порог, максимизирующий F1 на пуле (подбирается на VAL)."""
    yt = np.concatenate(trues)
    pp = np.concatenate(probs)
    grid = np.linspace(0.01, 0.99, 99)
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        f1 = M.f1_score_bin(yt, pp, thr=float(thr))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def _model_report(trues, probs) -> tuple[dict, dict]:
    pooled_t = np.concatenate(trues)
    pooled_p = np.concatenate(probs)
    point = M.compute_all(pooled_t, pooled_p)
    cis = {m: M.bootstrap_ci(trues, probs, m, n_boot=1000, seed=42) for m in METRIC_ORDER}
    return point, cis


def _fmt(v: float) -> str:
    return "n/a" if v != v else f"{v:.4f}"


def _fmt_ci(ci: tuple[float, float]) -> str:
    lo, hi = ci
    if lo != lo:
        return "n/a"
    return f"[{lo:.4f}, {hi:.4f}]"


def run_eval(unet_config: str, baseline_config: str, out_path: str) -> int:
    ucfg = cfgmod.load_yaml(unet_config)
    bcfg = cfgmod.load_yaml(baseline_config)

    print("[eval] инференс U-Net на test…")
    ut, up = _unet_predictions(ucfg, "test")
    print("[eval] инференс baseline на test…")
    bt, bp = _baseline_predictions(bcfg, "test")

    u_point, u_ci = _model_report(ut, up)
    b_point, b_ci = _model_report(bt, bp)

    # Диагностика порога: порог подбирается на VAL, метрики считаются на TEST.
    print("[eval] подбор порога по val…")
    u_val_t, u_val_p = _unet_predictions(ucfg, "val")
    b_val_t, b_val_p = _baseline_predictions(bcfg, "val")
    u_thr = _best_f1_threshold(u_val_t, u_val_p)
    b_thr = _best_f1_threshold(b_val_t, b_val_p)
    ut_pool, up_pool = np.concatenate(ut), np.concatenate(up)
    bt_pool, bp_pool = np.concatenate(bt), np.concatenate(bp)
    thr_diag = {
        "unet": (u_thr, M.iou_score(ut_pool, up_pool, u_thr), M.f1_score_bin(ut_pool, up_pool, u_thr)),
        "baseline": (b_thr, M.iou_score(bt_pool, bp_pool, b_thr), M.f1_score_bin(bt_pool, bp_pool, b_thr)),
    }

    # M-2: U-Net > baseline по IoU и F1 и PR-AUC, CI не пересекаются.
    m2_rows = []
    m2_pass = True
    for m in M2_METRICS:
        better = u_point[m] > b_point[m]
        disjoint = not M.ci_overlap(u_ci[m], b_ci[m])
        ok = better and disjoint
        m2_pass = m2_pass and ok
        m2_rows.append((m, better, disjoint, ok))

    lines: list[str] = []
    lines.append("# Сравнение моделей v1 — U-Net vs baseline (FR-7)\n")
    lines.append("Датасет: `data/processed/v1` (Тулун-2019, 7 каналов рельефа). ")
    lines.append("Test-split: 32 тайла. Метрики — попиксельно по пулу test.\n")
    lines.append("Bootstrap-CI 95%: 1000 ресэмплов **по тайлам** (OQ-6); ")
    lines.append("тайл — независимая единица (пиксели внутри коррелированы).\n\n")

    lines.append("## Метрики\n")
    lines.append("| Метрика | U-Net | baseline | CI U-Net (95%) | CI baseline (95%) | CI непересек. |")
    lines.append("|---|---|---|---|---|---|")
    for m in METRIC_ORDER:
        disjoint = not M.ci_overlap(u_ci[m], b_ci[m])
        lines.append(
            f"| {m} | {_fmt(u_point[m])} | {_fmt(b_point[m])} | "
            f"{_fmt_ci(u_ci[m])} | {_fmt_ci(b_ci[m])} | {'yes' if disjoint else 'no'} |"
        )
    lines.append("\n_Brier — меньше лучше (калибровка); остальные — больше лучше._\n")
    lines.append("_Метрики выше — при пороге 0.5._\n")

    lines.append("\n## Диагностика порога\n")
    lines.append(
        "ROC-AUC высок при низком IoU/F1 — модель **ранжирует** затопление хорошо, "
        "но порог 0.5 не оптимален для редкого класса (flood ~1.2%). Порог подобран "
        "по **val** (максимум F1), метрики — на **test**.\n"
    )
    lines.append("\n| Модель | Порог* (val) | IoU@0.5 | IoU@порог* | F1@0.5 | F1@порог* |")
    lines.append("|---|---|---|---|---|---|")
    for name, key in (("U-Net", "unet"), ("baseline", "baseline")):
        thr, iou_t, f1_t = thr_diag[key]
        base_point = u_point if key == "unet" else b_point
        lines.append(
            f"| {name} | {thr:.2f} | {_fmt(base_point['IoU'])} | {_fmt(iou_t)} | "
            f"{_fmt(base_point['F1'])} | {_fmt(f1_t)} |"
        )
    lines.append("")

    lines.append("\n## Критерий M-2 (§15)\n")
    lines.append("U-Net превосходит baseline по **IoU и F1 и PR-AUC**, CI 95% не пересекаются.\n")
    lines.append("\n| Метрика | U-Net > baseline | CI непересек. | M-2 по метрике |")
    lines.append("|---|---|---|---|")
    for m, better, disjoint, ok in m2_rows:
        lines.append(
            f"| {m} | {'yes' if better else 'no'} | {'yes' if disjoint else 'no'} | "
            f"{'✅ pass' if ok else '❌ fail'} |"
        )
    verdict = "✅ **M-2: PASS**" if m2_pass else "❌ **M-2: FAIL**"
    lines.append(f"\n### Итог: {verdict}\n")
    if not m2_pass:
        lines.append(
            "\n**План при fail (валидный научный результат):** датасет = одно событие "
            "(Тулун-2019), сильный дисбаланс flood ~1.2%. Направления: (1) больше событий/"
            "регионов в обучении; (2) тюнинг loss (Focal, pos_weight) и порога классификации; "
            "(3) увеличение числа эпох/энкодера; (4) измерение OQ-2 (качество лейблов S1 vs "
            "Copernicus EMS). U-Net недополучает данных для перевеса над попиксельным RF.\n"
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] отчёт → {out}  (M-2: {'PASS' if m2_pass else 'FAIL'})")
    return 0
