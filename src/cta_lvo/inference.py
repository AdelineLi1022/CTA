"""Prepare unlabeled CTA inputs; inference never consumes LVO annotations."""
from pathlib import Path
import nibabel as nib
import numpy as np

from .data import safe_id, read_json, write_json
from .geometry import load_case, roi_bounds


def prepare_inference(cases, source_task, output, roi='full', margin_mm=20.):
    source_task, output = Path(source_task), Path(output)
    if output.name != source_task.name:
        raise ValueError("Keep original TaskNNN_Name; place inference task under a separate parent directory")
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("Negative margin")
    original = read_json(source_task / 'proxy_metadata.json')
    if (original['roi'] == 'full') != (roi == 'full'):
        raise ValueError("Inference ROI mode must match training (CoW oracle training may use automatic CoW at inference)")
    if roi != 'full' and original.get('margin_mm') != margin_mm:
        raise ValueError("Use the same physical CoW margin as training")
    if len({c['patient_id'] for c in cases}) != len(cases) or not cases:
        raise ValueError("Inference requires unique nonempty cases")
    output.mkdir(parents=True, exist_ok=False)
    folder = output / 'raw_splitted' / 'imagesTs'
    folder.mkdir(parents=True)
    metadata = []
    for case in cases:
        pid = safe_id(case['patient_id'])
        if '.' in pid:
            raise ValueError("nnDetection IDs cannot contain periods")
        if roi == 'cow-auto' and case.get('cow_source') != 'automatic':
            raise ValueError("CoW inference requires cow_source='automatic' and an external predicted CoW mask")
        # Explicitly ignore LVO labels even when supplied in an existing manifest.
        clean = {**case, 'annotation_status': 'unknown', 'lvo': None}
        if roi == 'full':
            clean['cow'] = None
        images = load_case(clean)
        lo, hi = roi_bounds(images, roi, margin_mm)
        slices = tuple(slice(int(a), int(b)) for a,b in zip(lo, hi))
        nib.save(images['cta'].slicer[slices], folder / f'{pid}_0000.nii.gz')
        metadata.append({'patient_id': pid, 'roi_origin_voxel': lo.tolist()})
    dataset = read_json(source_task / 'dataset.json')
    dataset['test_labels'] = False
    write_json(output / 'dataset.json', dataset)
    write_json(output / 'inference_metadata.json', {'roi': roi, 'margin_mm': margin_mm, 'cases': metadata})
    return {'patients': len(cases), 'output': str(output)}
