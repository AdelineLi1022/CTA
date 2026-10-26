import argparse
import json
from pathlib import Path

from .data import inspect_parquet, import_parquet, read_json, write_json, split_cases
from .geometry import audit_case
from .export import export_dataset
from .evaluation import evaluate


def manifest_cases(path):
    path = Path(path).resolve()
    cases = read_json(path)["cases"]
    for case in cases:
        for key in ("cta", "lvo", "cow"):
            if case.get(key):
                p = Path(case[key])
                case[key] = str(p if p.is_absolute() else path.parent / p)
    return cases


def main():
    parser = argparse.ArgumentParser(description="ISLES24 sparse LVO localization research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download")
    p.add_argument("repo_id")
    p.add_argument("--revision", required=True, help="Pin a HF dataset commit for reproducibility")
    p.add_argument("--output", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("parquet_dir")
    p.add_argument("--output", required=True)
    p = sub.add_parser("import-parquet")
    p.add_argument("parquet_dir")
    p.add_argument("--mapping", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("audit")
    p.add_argument("manifest")
    p.add_argument("--output", required=True)
    p = sub.add_parser("split")
    p.add_argument("manifest")
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--output", required=True)
    p = sub.add_parser("export")
    p.add_argument("manifest")
    p.add_argument("--splits", required=True)
    p.add_argument("--output", required=True, help="New nnDetection TaskNNN_Name directory")
    p.add_argument("--vessel-labels", help="Optional manually reviewed BA/TICA/M1 component assignments")
    p.add_argument("--box-mm", type=float, default=10.)
    p.add_argument("--roi", choices=["full", "cow-oracle", "cow-auto"], default="full")
    p.add_argument("--margin-mm", type=float, default=20.)
    p = sub.add_parser("inference-input")
    p.add_argument("manifest")
    p.add_argument("--source-task", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--roi", choices=["full", "cow-auto"], default="full")
    p.add_argument("--margin-mm", type=float, default=20.)
    p = sub.add_parser("nndet")
    p.add_argument("stage", choices=["prep", "unpack", "train", "predict"])
    p.add_argument("task_dir")
    p.add_argument("--models", required=True)
    p.add_argument("--model", default="RetinaUNetV001_D3V001_3d")
    p.add_argument("--plan", default="D3V001_3d")
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("convert-predictions")
    p.add_argument("task_dir")
    p.add_argument("predictions_dir")
    p.add_argument("--trusted-local", action="store_true")
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("audit")
    p.add_argument("predictions", help="JSON list of {patient_id,center_mm:[R,A,S],score}")
    p.add_argument("--splits", required=True)
    p.add_argument("--subset", choices=["val", "test"], default="test")
    p.add_argument("--distance-mm", type=float, default=10.)
    p.add_argument("--vessel-labels", help="Reviewed segment assignments for classification evaluation")
    p.add_argument("--froc-thresholds", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--score-threshold", type=float, default=.5)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "download":
        from huggingface_hub import snapshot_download
        result = {"path": snapshot_download(repo_id=args.repo_id, repo_type="dataset",
                  revision=args.revision, local_dir=args.output, allow_patterns=["*.parquet", "*.json", "README.md"])}
    elif args.command == "inspect":
        result = inspect_parquet(args.parquet_dir, args.output)
    elif args.command == "import-parquet":
        result = import_parquet(args.parquet_dir, args.mapping, args.output)
    elif args.command == "audit":
        reports = []
        for case in manifest_cases(args.manifest):
            reports.append(audit_case(case))
        result = {"cases": reports, "patients": len(reports),
                  "positive": sum(c["annotation_status"] == "positive" for c in reports),
                  "negative": sum(c["annotation_status"] == "negative" for c in reports),
                  "unknown": sum(c["annotation_status"] == "unknown" for c in reports)}
        write_json(args.output, result)
    elif args.command == "split":
        result = split_cases(manifest_cases(args.manifest), args.seed)
        write_json(args.output, result)
    elif args.command == "export":
        result = export_dataset(manifest_cases(args.manifest), read_json(args.splits), args.output,
                                args.box_mm, args.roi, args.margin_mm,
                                read_json(args.vessel_labels) if args.vessel_labels else None)
    elif args.command == "inference-input":
        from .inference import prepare_inference
        result = prepare_inference(manifest_cases(args.manifest), args.source_task, args.output, args.roi, args.margin_mm)
    elif args.command == "nndet":
        from .nndetection import run_stage
        result = run_stage(args.task_dir, args.models, args.stage, args.model, args.workers, args.plan)
    elif args.command == "convert-predictions":
        from .nndetection import convert_predictions
        result = convert_predictions(args.task_dir, args.predictions_dir, args.output, args.trusted_local)
    else:
        ids = read_json(args.splits)[args.subset]
        cases = read_json(args.audit)["cases"]
        if len(set(ids)) != len(ids) or not set(ids) <= {c["patient_id"] for c in cases}:
            raise ValueError("Invalid evaluation split")
        cohort = [c for c in cases if c["patient_id"] in ids]
        if args.vessel_labels:
            from .classes import reviewed_segment
            reviews = read_json(args.vessel_labels)
            for case in cohort:
                for idx, target in enumerate(case["targets"], 1):
                    target["segment"] = reviewed_segment(reviews, case["patient_id"], idx)
        predictions = read_json(args.predictions)
        result = evaluate(cohort, predictions, args.distance_mm, args.score_threshold)
        result["froc_operating_points"] = [evaluate(cohort, predictions, args.distance_mm, float(t))
                                           for t in args.froc_thresholds.split(",")]
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
