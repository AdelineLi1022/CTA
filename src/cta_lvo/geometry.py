import numpy as np
import nibabel as nib
from scipy import ndimage


def load_case(case):
    if case.get("space") != "ncct":
        raise ValueError("Use explicitly verified NCCT-space CTA and masks (space='ncct')")
    status = case.get("annotation_status")
    if status not in {"positive", "negative", "unknown"}:
        raise ValueError("annotation_status must be positive, negative or unknown")
    cta = nib.load(case["cta"])
    if len(cta.shape) != 3 or cta.header.get_xyzt_units()[0] != "mm":
        raise ValueError("CTA must be 3D with physical units explicitly set to mm")
    if not np.isfinite(cta.affine).all() or abs(np.linalg.det(cta.affine[:3, :3])) < 1e-10:
        raise ValueError("Invalid CTA affine")
    images = {"cta": cta}
    for key in ("lvo", "cow"):
        if case.get(key):
            img = nib.load(case[key])
            if img.shape != cta.shape or not np.allclose(img.affine, cta.affine, rtol=0, atol=1e-3):
                raise ValueError(f"{key} grid differs from CTA; obtain registered images, do not copy affines")
            if img.header.get_xyzt_units()[0] != "mm":
                raise ValueError(f"{key} units must be mm")
            data = img.get_fdata()
            if not np.isfinite(data).all() or (data < 0).any() or not np.equal(data, np.round(data)).all():
                raise ValueError(f"{key} must have finite nonnegative integer labels")
            images[key] = img
    if not np.isfinite(cta.get_fdata()).all():
        raise ValueError("CTA contains nonfinite intensities")
    present = "lvo" in images and np.any(images["lvo"].get_fdata() > 0)
    if status == "positive" and not present:
        raise ValueError("Positive case requires nonempty sparse LVO mask")
    if status == "negative" and present:
        raise ValueError("Negative case has nonempty LVO mask")
    return images


def targets(images):
    if "lvo" not in images:
        return []
    labels, count = ndimage.label(images["lvo"].get_fdata() > 0,
                                   structure=np.ones((3, 3, 3)))
    affine = images["cta"].affine
    out = []
    for index in range(1, count + 1):
        voxels = np.argwhere(labels == index)
        center = voxels.mean(axis=0)
        world = nib.affines.apply_affine(affine, center)
        out.append({"center_voxel": center.tolist(), "center_mm": world.tolist(),
                    "annotation_voxels": len(voxels)})
    return out


def roi_bounds(images, mode, margin_mm):
    shape = np.array(images["cta"].shape)
    if mode == "full":
        return np.zeros(3, dtype=int), shape
    if "cow" not in images:
        raise ValueError("CoW ROI requires a CoW mask")
    coords = np.argwhere(images["cow"].get_fdata() > 0)
    if not len(coords):
        raise ValueError("CoW mask is empty")
    inverse = np.linalg.inv(images["cta"].affine[:3, :3])
    margin = np.ceil(margin_mm * np.linalg.norm(inverse, axis=1)).astype(int)
    return np.maximum(0, coords.min(0)-margin), np.minimum(shape, coords.max(0)+1+margin)


def audit_case(case):
    images = load_case(case)
    points = targets(images)
    result = {"patient_id": case["patient_id"], "annotation_status": case["annotation_status"],
              "shape": list(images["cta"].shape),
              "spacing_mm": nib.affines.voxel_sizes(images["cta"].affine).tolist(),
              "lvo_components": len(points), "targets": points, "cow_labels": {}}
    if "cow" in images:
        cow = images["cow"].get_fdata()
        labels, counts = np.unique(cow, return_counts=True)
        result["cow_labels"] = {str(int(k)): int(v) for k, v in zip(labels, counts)}
        foreground = np.argwhere(cow > 0)
        if len(foreground):
            from scipy.spatial import cKDTree
            tree = cKDTree(nib.affines.apply_affine(images["cta"].affine, foreground))
            for point in points:
                distance, idx = tree.query(point["center_mm"])
                nearest = foreground[idx]
                point["nearest_cow_distance_mm"] = float(distance)
                point["nearest_cow_label_candidate"] = int(cow[tuple(nearest)])
    return result
