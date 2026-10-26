from pathlib import Path
import pickle
import re
import nibabel as nib
import numpy as np

from .data import safe_id, write_json
from .classes import SEGMENTS, reviewed_segment
from .geometry import load_case, targets, roi_bounds


def export_dataset(cases, splits, output, box_mm=10., roi="full", margin_mm=20., vessel_labels=None):
    """nnDetection raw data: instance masks encode proxy boxes, never thrombus extent."""
    if not np.isfinite(box_mm) or box_mm <= 0 or not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("Box size must be positive and margin nonnegative")
    ids = [c["patient_id"] for c in cases]
    assigned = [p for values in splits.values() for p in values]
    if set(splits) != {"train", "val", "test"} or len(set(assigned)) != len(assigned) or set(ids) != set(assigned):
        raise ValueError("Splits must assign every unique patient exactly once")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate patients")
    output = Path(output)
    if not re.fullmatch(r"Task[0-9]{3}_[A-Za-z0-9_-]+", output.name):
        raise ValueError("Output must be a TaskNNN_Name directory, e.g. Task100_ISLES24LVO")
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw_splitted"
    raw.mkdir()
    metadata, included = [], set()
    for case in cases:
        pid = safe_id(case["patient_id"])
        if "." in pid:
            raise ValueError("nnDetection patient IDs cannot contain periods")
        if case["annotation_status"] == "unknown":
            metadata.append({"patient_id": pid, "excluded": "unknown annotation status"})
            continue
        if roi == "cow-auto" and case.get("cow_source") != "automatic":
            raise ValueError("cow-auto requires cow_source=automatic and a predicted CoW mask")
        images = load_case(case)
        points = targets(images)
        lo, hi = roi_bounds(images, roi, margin_mm)
        shape = hi - lo
        centers = [np.array(t["center_voxel"]) for t in points]
        # Do not enlarge a CoW crop using LVO ground truth: that leaks the target.
        if any(np.any(c < lo) or np.any(c >= hi) for c in centers):
            raise ValueError(f"{pid}: CoW ROI misses LVO center; report ROI coverage before training")
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        cta = images["cta"]
        crop = cta.slicer[slices]
        seg = np.zeros(tuple(shape), dtype=np.uint16)
        half = box_mm / (2 * nib.affines.voxel_sizes(cta.affine))
        instances, boxes = {}, []
        if len(centers) > 65535:
            raise ValueError("Too many instances")
        for idx, center in enumerate(centers, 1):
            c = center-lo
            lower = np.maximum(0, np.floor(c-half).astype(int))
            upper = np.minimum(shape, np.ceil(c+half).astype(int)+1)
            region = tuple(slice(int(a), int(b)) for a,b in zip(lower, upper))
            if np.any(seg[region]):
                raise ValueError(f"{pid}: overlapping proxy boxes; manually review separate components")
            seg[region] = idx
            segment = reviewed_segment(vessel_labels, pid, idx) if vessel_labels is not None else None
            instances[str(idx)] = SEGMENTS[segment] if segment else 0
            if segment:
                points[idx-1]["segment"] = segment
            boxes.append({"instance": idx, "lo_inclusive": lower.tolist(), "hi_exclusive": upper.tolist()})
        heldout = pid in splits["test"]
        image_dir = raw / ("imagesTs" if heldout else "imagesTr")
        label_dir = raw / ("labelsTs" if heldout else "labelsTr")
        image_dir.mkdir(exist_ok=True)
        label_dir.mkdir(exist_ok=True)
        nib.save(crop, image_dir / f"{pid}_0000.nii.gz")
        header = crop.header.copy()
        header.set_data_dtype(np.uint16)
        nib.save(nib.Nifti1Image(seg, crop.affine, header), label_dir / f"{pid}.nii.gz")
        write_json(label_dir / f"{pid}.json", {"instances": instances})
        included.add(pid)
        metadata.append({"patient_id": pid, "roi_origin_voxel": lo.tolist(),
                         "targets": points, "boxes": boxes, "roi": roi})
    write_json(output / "dataset.json", {"name": "ISLES24_LVO", "task": output.name,
               "test_labels": bool(splits["test"]), "labels": ({str(v): k for k,v in SEGMENTS.items()} if vessel_labels is not None else {"0": "LVO"}),
               "modalities": {"0": "CT"}, "dim": 3})
    # nnDetection uses one predefined train/val fold; test cases stay completely separate.
    fold = [{k: sorted(p for p in splits[k] if p in included) for k in ("train", "val")}]
    write_json(output / "splits_final.json", fold)
    # Upstream BaseModule reads preprocessed/splits_final.pkl, not JSON.
    preprocessed = output / "preprocessed"
    preprocessed.mkdir()
    with (preprocessed / "splits_final.pkl").open("wb") as stream:
        pickle.dump(fold, stream, protocol=4)
    write_json(output / "proxy_metadata.json", {"box_mm": box_mm, "roi": roi, "margin_mm": margin_mm,
               "warning": "Sparse-location proxy boxes; not thrombus boundaries. CoW GT ROI is an oracle experiment.",
               "cases": metadata})
    return {"exported": len(included), "excluded": len(cases)-len(included), "output": str(output)}
