"""Explicit Parquet mapping; never infer registration or negative labels."""
import gzip
import hashlib
import json
import re
import tempfile

import numpy as np
from pathlib import Path

import nibabel as nib
import pyarrow.parquet as pq


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_id(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe patient identifier: {value!r}")
    return value


def nested(row, key):
    if isinstance(key, dict) and set(key) == {"constant"}:
        return key["constant"]
    for part in key.split("."):
        row = row[part]
    return row


def parquet_files(root, pattern="*.parquet"):
    # Nonrecursive by default: environments under the data root contain fixture Parquet files.
    files = sorted(p for p in Path(root).glob(pattern) if p.is_file())
    if not files:
        raise ValueError(f"No Parquet shards match {pattern!r} in {root}")
    return files


def inspect_parquet(root, output, pattern="*.parquet"):
    files = parquet_files(root, pattern)
    if not files:
        raise ValueError("No Parquet shards found")
    report = []
    for file in files:
        p = pq.ParquetFile(file)
        report.append({"file": str(file), "rows": p.metadata.num_rows,
                       "schema": str(p.schema_arrow)})
    write_json(output, report)
    return {"shards": len(files), "rows": sum(x["rows"] for x in report)}


def materialize(value, destination, source_root):
    """Accept NIfTI paths, raw bytes, or HF {bytes,path} structs only."""
    if isinstance(value, dict):
        value = value.get("bytes") if value.get("bytes") is not None else value.get("path")
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = source_root / path
        image = nib.load(path)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        if payload[:2] == b"\x1f\x8b":
            payload = gzip.decompress(payload)
        # NIfTI-1 and NIfTI-2 are both supported, without pickle/deserialization.
        try:
            image = nib.Nifti1Image.from_bytes(payload)
        except Exception:
            image = nib.Nifti2Image.from_bytes(payload)
    else:
        raise ValueError("Expected NIfTI path or HF bytes/path struct; arrays/archives need explicit conversion")
    nib.save(image, destination)


def import_parquet(root, mapping_file, output, pattern="*.parquet"):
    root, output = Path(root).resolve(), Path(output).resolve()
    mapping = read_json(mapping_file)
    required = {"patient_id", "cta", "lvo", "cow", "space", "annotation_status"}
    if set(mapping) != required:
        raise ValueError(f"Mapping must contain exactly {sorted(required)}")
    shards = parquet_files(root, pattern)
    if not shards:
        raise ValueError("No Parquet shards found")
    # Do not overwrite an existing import, even after a partial failure.
    output.mkdir(parents=True, exist_ok=False)
    cases, seen = [], set()
    for shard in shards:
        columns = sorted({v.split(".")[0] for v in mapping.values() if isinstance(v, str)})
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1, columns=columns):
            for row in batch.to_pylist():
                pid = safe_id(nested(row, mapping["patient_id"]))
                if pid in seen:
                    raise ValueError(f"Duplicate patient {pid}; consolidate repeated acquisitions explicitly")
                seen.add(pid)
                case = {"patient_id": pid}
                for field in ("space", "annotation_status"):
                    case[field] = nested(row, mapping[field])
                folder = output / pid
                folder.mkdir()
                for field in ("cta", "lvo", "cow"):
                    value = nested(row, mapping[field])
                    case[field] = None
                    if value is not None:
                        target = folder / f"{field}.nii.gz"
                        materialize(value, target, root)
                        case[field] = str(target)
                cases.append(case)
    write_json(output / "manifest.json", {"cases": cases})
    return {"patients": len(cases), "manifest": str(output / "manifest.json")}


def split_cases(cases, seed=24):
    """Patient-level, status-stratified deterministic 70/15/15 assignment."""
    if len({c["patient_id"] for c in cases}) != len(cases):
        raise ValueError("Patient IDs must be unique")
    splits = {"train": [], "val": [], "test": []}
    groups = {}
    for c in cases:
        groups.setdefault(c["annotation_status"], []).append(c["patient_id"])
    for ids in groups.values():
        ids.sort(key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())
        n = len(ids)
        # Small cohorts can have empty held-out sets; report rather than leak patients.
        ntest = max(1, round(n * .15)) if n >= 3 else 0
        nval = max(1, round(n * .15)) if n >= 3 else 0
        splits["test"].extend(ids[:ntest])
        splits["val"].extend(ids[ntest:ntest+nval])
        splits["train"].extend(ids[ntest+nval:])
    return {k: sorted(v) for k, v in splits.items()}


def probe_isles(root, output, pattern="train-*.parquet", limit=1):
    """Inspect embedded NIfTI geometry without declaring registration or negatives."""
    if limit < 1:
        raise ValueError("limit must be positive")
    root = Path(root).resolve()
    reports = []
    fields = ("ncct", "cta", "lvo_mask", "cow_segmentation")
    with tempfile.TemporaryDirectory(prefix="cta_probe_") as temporary:
        for shard in parquet_files(root, pattern):
            parquet = pq.ParquetFile(shard)
            columns = [k for k in ("subject_id", *fields) if k in parquet.schema_arrow.names]
            for batch in parquet.iter_batches(batch_size=1, columns=columns):
                for row in batch.to_pylist():
                    report = {"file": str(shard), "subject_id": row.get("subject_id"), "images": {}}
                    grids = {}
                    for field in fields:
                        value = row.get(field)
                        if value is None or (isinstance(value, dict) and value.get("bytes") is None and not value.get("path")):
                            report["images"][field] = {"present": False}
                            continue
                        detail = {"present": True, "embedded_path": value.get("path") if isinstance(value, dict) else None}
                        try:
                            target = Path(temporary) / f"{field}.nii.gz"
                            materialize(value, target, root)
                            img = nib.load(target)
                            detail.update(shape=list(img.shape), spacing=nib.affines.voxel_sizes(img.affine).tolist(),
                                          affine=img.affine.tolist(), units=img.header.get_xyzt_units()[0])
                            grids[field] = (img.shape, img.affine.copy())
                            if field in ("lvo_mask", "cow_segmentation"):
                                data = img.get_fdata()
                                labels, counts = np.unique(data, return_counts=True)
                                detail["nonzero_voxels"] = int(np.count_nonzero(data))
                                detail["label_counts"] = {str(float(k)): int(v) for k,v in zip(labels[:100], counts[:100])}
                                detail["unique_labels"] = len(labels)
                            del img
                            target.unlink()
                        except Exception as exc:
                            detail["error"] = f"{type(exc).__name__}: {exc}"
                        report["images"][field] = detail
                    report["same_grid_as_ncct"] = {
                        key: shape == grids["ncct"][0] and bool(np.allclose(affine, grids["ncct"][1], rtol=0, atol=1e-3))
                        for key, (shape, affine) in grids.items() if key != "ncct" and "ncct" in grids
                    }
                    reports.append(report)
                    if len(reports) >= limit:
                        write_json(output, reports)
                        return reports
    write_json(output, reports)
    return reports
