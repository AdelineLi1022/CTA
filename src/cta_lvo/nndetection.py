"""nnDetection command integration and restored-box coordinate conversion.

Reference: MIC-DKFZ/nnDetection 97a58f3110b71caf1b4bcc1851e67cf11e987fc5.
"""
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from .data import read_json, write_json


def run_stage(task_dir, models, stage, model="RetinaUNetV001_D3V001_3d", workers=4, plan="D3V001_3d"):
    task_dir = Path(task_dir).resolve()
    if not (task_dir / "dataset.json").is_file():
        raise ValueError("Expected exported task directory")
    if workers < 1:
        raise ValueError("workers must be positive")
    if stage != "predict":
        fold = read_json(task_dir / "splits_final.json")[0]
        if not fold["train"] or not fold["val"]:
            raise ValueError("Training and validation must both be nonempty")
    task = task_dir.name[4:7]
    commands = {
        "prep": ["nndet_prep", task, "--full_check", "-np", str(workers), "-npp", str(workers)],
        "unpack": ["nndet_unpack", str(task_dir / "preprocessed" / plan / "imagesTr"), str(workers)],
        "train": ["nndet_train", task, "-o", "exp.fold=0", "--sweep"],
        "predict": ["nndet_predict", task, model, "--fold", "0"],
    }
    command = commands[stage]
    if not shutil.which(command[0]):
        raise RuntimeError("nnDetection is not installed in this interpreter's environment; see README GPU setup")
    environment = os.environ.copy()
    environment.update(det_data=str(task_dir.parent), det_models=str(Path(models).resolve()),
                       OMP_NUM_THREADS="1", det_num_threads=str(workers))
    Path(models).mkdir(parents=True, exist_ok=True)
    # Fail before expensive planning if CUDA/custom operators are unavailable.
    if stage != "unpack":
        subprocess.run(
            [sys.executable, "-c",
             "import torch; import nndet._C; assert torch.cuda.is_available(), 'CUDA GPU required'"],
            env=environment, check=True,
        )
    subprocess.run(command, env=environment, check=True)
    return {"stage": stage, "command": command, "completed": True}


def convert_predictions(task_dir, predictions_dir, output, trusted_local=False):
    """Read only locally generated nnDetection pickle results after explicit opt-in."""
    if not trusted_local:
        raise ValueError("Pickle can execute code; only opt in for your own locally generated results")
    task_dir, predictions_dir = Path(task_dir), Path(predictions_dir)
    result, seen = [], set()
    class_names = read_json(task_dir / "dataset.json")["labels"]
    files = sorted(predictions_dir.glob("*_boxes.pkl"))
    if not files:
        raise ValueError("No *_boxes.pkl predictions found")
    for path in files:
        pid = path.name[:-len("_boxes.pkl")]
        image_paths = [p for folder in ("imagesTr", "imagesTs")
                       if (p := task_dir / "raw_splitted" / folder / f"{pid}_0000.nii.gz").is_file()]
        if len(image_paths) != 1 or pid in seen:
            raise ValueError(f"Missing or ambiguous reference image for {pid}")
        seen.add(pid)
        with path.open("rb") as stream:
            pred = pickle.load(stream)
        if not pred.get("restore", False):
            raise ValueError("Require restored predictions, not preprocessed-space boxes")
        boxes = np.asarray(pred["pred_boxes"], dtype=float).reshape(-1, 6)
        scores = np.asarray(pred["pred_scores"], dtype=float)
        labels = np.asarray(pred["pred_labels"])
        if scores.shape != (len(boxes),) or labels.shape != (len(boxes),):
            raise ValueError("Prediction array lengths differ")
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
            raise ValueError("Invalid prediction values")
        if (boxes[:, [2,3,5]] < boxes[:, [0,1,4]]).any() or any(str(int(label)) not in class_names or int(label) != label for label in labels):
            raise ValueError("Invalid boxes or non-LVO classes")
        # nnDetection: SimpleITK array axes z,y,x, box order a1,b1,a2,b2,c1,c2.
        # Nibabel: x,y,z. The exported crop affine already includes ROI offset.
        centers_zyx = (boxes[:, [0,1,4]] + boxes[:, [2,3,5]]) / 2
        image = nib.load(image_paths[0])
        centers_mm = nib.affines.apply_affine(image.affine, centers_zyx[:, ::-1])
        result.extend({"patient_id": pid, "center_mm": center.tolist(), "score": float(score), "segment": class_names[str(int(label))]}
                      for center, score, label in zip(centers_mm, scores, labels))
    write_json(output, result)
    return {"patients": len(seen), "predictions": len(result), "output": str(output)}
