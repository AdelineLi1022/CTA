"""Reviewed segment assignments only; nearest CoW labels are never ground truth."""
SEGMENTS = {"BA": 0, "TICA": 1, "M1": 2}


def reviewed_segment(reviews, patient_id, component):
    try:
        entry = reviews[patient_id][str(component)]
    except KeyError as exc:
        raise ValueError(f"Missing reviewed vessel assignment: {patient_id} component {component}") from exc
    if entry.get("reviewed") is not True or entry.get("segment") not in SEGMENTS:
        raise ValueError(f"Unreviewed/unsupported segment: {patient_id} component {component}")
    return entry["segment"]
