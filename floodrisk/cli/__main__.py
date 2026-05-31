"""CLI entry point: `python -m floodrisk <subcommand>`.

Реализованы: ``data`` (Этап 1, FR-1…FR-4), ``train``/``eval`` (Этап 2, FR-5…FR-7),
``infer`` (Этап 3, FR-8). ``explain`` — заглушка до Этапа 5 (SRS §17, §10.1).
"""

import argparse
import contextlib
import sys
from pathlib import Path


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


def _run_infer(args) -> int:
    from sqlmodel import Session

    from floodrisk.db.session import create_db_and_tables, engine
    from floodrisk.inference import service

    create_db_and_tables()
    out = Path(args.out)
    with Session(engine) as session:
        try:
            res = service.predict(
                session,
                args.bbox,
                args.scenario,
                args.model,
                run_dir=out,
                persist=args.save_to_db,
            )
        except service.InvalidBBox as exc:
            print(f"[infer] invalid_bbox: {exc}", file=sys.stderr)
            return 3
        except service.OutOfCoverage as exc:
            print(
                f"[infer] bbox_out_of_coverage; coverage_wgs84={exc.coverage_wgs84}",
                file=sys.stderr,
            )
            return 22
        except (service.UnknownScenario, service.UnknownModel) as exc:
            print(f"[infer] {exc}: available={exc.available}", file=sys.stderr)
            return 4
    print(f"[infer] run_id={res['run_id']} -> {out}")
    print(f"[infer] aggregates={res['aggregates']}")
    return 0


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

    inf = sub.add_parser("infer", help="CLI-инференс (Этап 3, FR-8)")
    inf.add_argument("--bbox", type=_parse_bbox, required=True, help="bbox 'W,S,E,N' (EPSG:4326)")
    inf.add_argument("--scenario", default="p100y", help="scenario_id (p1y/p10y/p100y)")
    inf.add_argument("--model", default="unet-v1", help="version-имя модели")
    inf.add_argument("--out", default="runs/cli", help="каталог для артефактов")
    inf.add_argument(
        "--save-to-db", action="store_true", help="записать запуск в БД (как делает API)"
    )
    inf.set_defaults(func=_run_infer)

    sub.add_parser("explain", help="объяснимость (Этап 5)")
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
