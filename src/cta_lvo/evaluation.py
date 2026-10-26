"""Physical-space, confidence-ranked one-to-one sparse-location evaluation."""
import numpy as np


def evaluate(ground_truth, predictions, distance_mm=10., score_threshold=0.5):
    if not np.isfinite(distance_mm) or distance_mm <= 0 or not 0 <= score_threshold <= 1:
        raise ValueError("Invalid distance or score threshold")
    gt = {}
    for case in ground_truth:
        pid = case["patient_id"]
        if pid in gt:
            raise ValueError("Duplicate ground truth patient")
        if case["annotation_status"] not in {"positive", "negative", "unknown"}:
            raise ValueError("Invalid annotation status")
        if case["annotation_status"] == "unknown":
            continue
        centers = np.asarray([t["center_mm"] for t in case["targets"]], dtype=float).reshape(-1, 3)
        if not np.isfinite(centers).all():
            raise ValueError("Invalid ground truth centers")
        if (case["annotation_status"] == "positive") != (len(centers) > 0):
            raise ValueError("Ground truth status/target mismatch")
        gt[pid] = (case["annotation_status"], centers)
    if not gt:
        raise ValueError("No evaluable patients")
    by_patient = {pid: [] for pid in gt}
    for pred in predictions:
        pid = pred["patient_id"]
        if pid not in gt:
            raise ValueError(f"Prediction patient {pid} is outside selected evaluation cohort")
        center = np.asarray(pred["center_mm"], dtype=float)
        score = float(pred["score"])
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Predictions require finite RAS mm centers and scores in [0,1]")
        if score >= score_threshold:
            by_patient[pid].append((score, center, pred.get("segment")))
    tp = fp = total = negatives = negative_clear = positive_detected = positive_cases = 0
    distances = []
    classification = {"matched_reviewed_targets": 0, "correct": 0, "confusion": {}}
    truth_lookup = {c["patient_id"]: c for c in ground_truth}
    for pid, (status, centers) in gt.items():
        matched = set()
        total += len(centers)
        detections = sorted(by_patient[pid], key=lambda x: -x[0])
        for _, center, predicted_segment in detections:
            candidates = [(float(np.linalg.norm(center-target)), i)
                          for i, target in enumerate(centers) if i not in matched]
            if candidates and min(candidates)[0] <= distance_mm:
                distance, i = min(candidates)
                matched.add(i)
                tp += 1
                distances.append(distance)
                segment = truth_lookup[pid]["targets"][i].get("segment")
                if segment is not None:
                    classification["matched_reviewed_targets"] += 1
                    classification["correct"] += predicted_segment == segment
                    pair = f"{segment}->{predicted_segment or 'missing'}"
                    classification["confusion"][pair] = classification["confusion"].get(pair, 0) + 1
            else:
                fp += 1
        if status == "negative":
            negatives += 1
            negative_clear += not detections
        else:
            positive_cases += 1
            positive_detected += bool(matched)
    classification["accuracy_on_localized_targets"] = (classification["correct"]/classification["matched_reviewed_targets"]
                                                        if classification["matched_reviewed_targets"] else None)
    return {"classification": classification, "patients": len(gt), "targets": total, "tp": tp, "fp": fp, "fn": total-tp,
            "score_threshold": score_threshold, "distance_mm": distance_mm,
            "localization_sensitivity": tp/total if total else None,
            "positive_case_localization_sensitivity": positive_detected/positive_cases if positive_cases else None,
            "fp_per_case": fp/len(gt), "verified_negative_cases": negatives,
            "negative_case_specificity": negative_clear/negatives if negatives else None,
            "matched_distance_mean_mm": float(np.mean(distances)) if distances else None}
