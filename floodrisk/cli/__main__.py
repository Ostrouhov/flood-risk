"""CLI entry point: `python -m floodrisk <subcommand>`.

Этап 1 реализует группу ``data`` (fetch/preprocess/label/manifest/verify, FR-1…FR-4).
Подкоманды train/eval/infer/explain — заглушки до соответствующих этапов (SRS §17, §10.1).
"""

import argparse
import contextlib
import sys


def _parse_bbox(value: str) -> list[float]:
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox должен быть 'W,S,E,N'")
    return parts


def _run_data(args) -> int:
    from floodrisk.data import pipeline
    from floodrisk.data.config import load_data_config

    cfg = load_data_config()
    if args.bbox is not None:
        cfg = cfg.model_copy(update={"pilot_bbox_wgs84": args.bbox})

    if args.action == "fetch":
        prov = pipeline.run_fetch(cfg, skip_era5=args.skip_era5)
        print(f"[data fetch] источников выгружено: {len(prov)}")
    elif args.action == "preprocess":
        out = pipeline.run_preprocess(cfg)
        print(f"[data preprocess] тайлы и index.parquet в {out}")
    elif args.action == "label":
        res = pipeline.run_label(cfg)
        print(f"[data label] маски затопления: {', '.join(res)}")
    elif args.action == "manifest":
        path = pipeline.run_manifest(cfg)
        print(f"[data manifest] {path}")
    elif args.action == "verify":
        return pipeline.run_verify(args.manifest)
    return 0


def _run_train(args) -> int:
    from floodrisk.ml.train import run_train

    return run_train(args.config)


# Сопоставление version-имён моделей (как в Makefile) с конфигами.
_MODEL_CONFIGS = {
    "unet-v1": "configs/unet_v1.yaml",
    "baseline-v1": "configs/baseline_v1.yaml",
}


def _run_eval(args) -> int:
    from floodrisk.ml.evaluate import run_eval

    names = [n.strip() for n in args.models.split(",")]
    cfgs = {n: _MODEL_CONFIGS.get(n, n) for n in names}
    unet_cfg = next((c for n, c in cfgs.items() if "unet" in n), "configs/unet_v1.yaml")
    base_cfg = next((c for n, c in cfgs.items() if "baseline" in n), "configs/baseline_v1.yaml")
    return run_eval(unet_cfg, base_cfg, args.out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="floodrisk", description="floodrisk CLI")
    sub = parser.add_subparsers(dest="cmd")

    data = sub.add_parser("data", help="data pipeline (Этап 1, FR-1…FR-4)")
    data.add_argument("action", choices=["fetch", "preprocess", "label", "manifest", "verify"])
    data.add_argument(
        "--bbox", type=_parse_bbox, default=None, help="переопределить bbox 'W,S,E,N'"
    )
    data.add_argument("--manifest", default=None, help="путь к manifest.yaml для verify")
    data.add_argument(
        "--skip-era5", action="store_true", help="не выгружать ERA5 (нет CDS_API_KEY)"
    )
    data.set_defaults(func=_run_data)

    train = sub.add_parser("train", help="обучение моделей (Этап 2, FR-5/FR-6)")
    train.add_argument("--config", required=True, help="путь к YAML-конфигу (unet/baseline)")
    train.set_defaults(func=_run_train)

    ev = sub.add_parser("eval", help="сравнение моделей (Этап 2, FR-7)")
    ev.add_argument("--models", default="unet-v1,baseline-v1", help="version-имена через запятую")
    ev.add_argument("--out", default="reports/comparison_v1.md", help="путь к отчёту")
    ev.set_defaults(func=_run_eval)

    for name, help_text in [
        ("infer", "CLI-инференс (Этап 3)"),
        ("explain", "объяснимость (Этап 5)"),
    ]:
        sub.add_parser(name, help=help_text)
    return parser


def main() -> int:
    # Windows-консоль по умолчанию cp1251 → юникод в выводе падает. Принудительно UTF-8.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8")

    parser = build_parser()
    args, _rest = parser.parse_known_args()

    if args.cmd is None:
        parser.print_help()
        return 0
    if hasattr(args, "func"):
        return args.func(args)

    print(
        f"[floodrisk cli] подкоманда '{args.cmd}' будет реализована на следующих этапах "
        "(см. docs/srs.md §17).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
