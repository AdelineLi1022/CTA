"""Explicit Parquet mapping; never infer registration or negative labels."""
import gzip
import hashlib
import json
import re
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


def inspect_parquet(root, output):
    files = sorted(Path(root).rglob("*.parquet"))
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


def import_parquet(root, mapping_file, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    mapping = read_json(mapping_file)
    required = {"patient_id", "cta", "lvo", "cow", "space", "annotation_status"}
    if set(mapping) != required:
        raise ValueError(f"Mapping must contain exactly {sorted(required)}")
    shards = sorted(root.rglob("*.parquet"))
    if not shards:
        raise ValueError("No Parquet shards found")
    # Do not overwrite an existing import, even after a partial failure.
    output.mkdir(parents=True, exist_ok=False)
    cases, seen = [], set()
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1):
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
