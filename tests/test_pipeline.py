import gzip
import json
import pickle
import subprocess
import sys

import nibabel as nib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cta_lvo.data import import_parquet, inspect_parquet, split_cases, write_json
from cta_lvo.geometry import audit_case, load_case, targets
from cta_lvo.export import export_dataset
from cta_lvo.evaluation import evaluate
from cta_lvo.nndetection import convert_predictions


def image(array, affine=None):
    if affine is None:
        affine = np.diag([2., 1., 3., 1.])
        affine[:3, 3] = [10, -20, 30]
    img = nib.Nifti1Image(array, affine)
    img.header.set_xyzt_units('mm')
    return img


@pytest.fixture
def case(tmp_path):
    cta = np.zeros((24, 24, 24), dtype=np.float32)
    lvo = np.zeros(cta.shape, dtype=np.uint8)
    lvo[10, 11, 12] = 1
    cow = np.zeros_like(lvo)
    cow[8:15, 8:15, 8:15] = 3
    result = {'patient_id': 'p001', 'space': 'ncct', 'annotation_status': 'positive'}
    for key, arr in [('cta', cta), ('lvo', lvo), ('cow', cow)]:
        path = tmp_path / f'{key}.nii.gz'
        nib.save(image(arr), path)
        result[key] = str(path)
    return result


def test_parquet_to_cli_end_to_end(case, tmp_path):
    row = {**case}
    for key in ('cta', 'lvo', 'cow'):
        row[key] = {'bytes': open(case[key], 'rb').read(), 'path': None}
    root = tmp_path / 'parquet'
    root.mkdir()
    pq.write_table(pa.Table.from_pylist([row]), root / 'data.parquet')
    assert inspect_parquet(root, tmp_path / 'schema.json') == {'shards': 1, 'rows': 1}
    mapping = tmp_path / 'mapping.json'
    write_json(mapping, {k: k for k in case})
    imported = tmp_path / 'imported'
    import_parquet(root, mapping, imported)
    manifest = imported / 'manifest.json'
    audit = tmp_path / 'audit.json'
    subprocess.run([sys.executable, '-m', 'cta_lvo.cli', 'audit', str(manifest),
                    '--output', str(audit)], check=True, capture_output=True)
    report = json.loads(audit.read_text())
    assert report['positive'] == 1
    point = report['cases'][0]['targets'][0]
    assert point['center_mm'] == [30., -9., 66.]
    assert point['nearest_cow_distance_mm'] == 0
    splits = {'train': ['p001'], 'val': [], 'test': []}
    task = tmp_path / 'Task100_LVO'
    export_dataset(json.loads(manifest.read_text())['cases'], splits, task, box_mm=4, roi='cow-oracle', margin_mm=2)
    seg = nib.load(task / 'raw_splitted/labelsTr/p001.nii.gz')
    cta = nib.load(task / 'raw_splitted/imagesTr/p001_0000.nii.gz')
    assert seg.shape == cta.shape
    assert np.max(seg.get_fdata()) == 1
    # Crop coordinate transforms remain in the original RAS physical space.
    meta = json.loads((task / 'proxy_metadata.json').read_text())['cases'][0]
    crop_center = np.array(point['center_voxel']) - meta['roi_origin_voxel']
    assert np.allclose(nib.affines.apply_affine(cta.affine, crop_center), point['center_mm'])
    with open(task / 'preprocessed/splits_final.pkl', 'rb') as f:
        assert pickle.load(f) == [{'train': ['p001'], 'val': []}]
    assert json.loads((task / 'dataset.json').read_text())['modalities'] == {'0': 'CT'}
    pred_dir = tmp_path / 'predictions'
    pred_dir.mkdir()
    z, y, x = crop_center[::-1]
    with open(pred_dir / 'p001_boxes.pkl', 'wb') as f:
        pickle.dump({'restore': True, 'pred_boxes': np.array([[z-1,y-1,z+1,y+1,x-1,x+1]]),
                     'pred_scores': np.array([.9]), 'pred_labels': np.array([0])}, f)
    output = tmp_path / 'pred.json'
    convert_predictions(task, pred_dir, output, trusted_local=True)
    pred = json.loads(output.read_text())
    assert pred[0]['center_mm'] == point['center_mm']
    metric = evaluate(report['cases'], pred)
    assert metric['tp'] == 1 and metric['fp'] == 0


def test_misaligned_mask_rejected(case):
    mask = nib.load(case['lvo'])
    affine = mask.affine.copy()
    affine[0, 3] += 1
    nib.save(image(mask.get_fdata(), affine), case['lvo'])
    with pytest.raises(ValueError, match='grid differs'):
        load_case(case)


def test_missing_annotation_not_negative(case):
    case['lvo'] = None
    with pytest.raises(ValueError, match='nonempty'):
        load_case(case)
    case['annotation_status'] = 'unknown'
    assert audit_case(case)['annotation_status'] == 'unknown'


def test_one_to_one_matching_and_negative_specificity():
    truth = [{'patient_id': 'p', 'annotation_status': 'positive', 'targets': [{'center_mm': [0,0,0]}]},
             {'patient_id': 'n', 'annotation_status': 'negative', 'targets': []}]
    predictions = [{'patient_id': 'p', 'center_mm': [0,0,1], 'score': .9},
                   {'patient_id': 'p', 'center_mm': [0,0,2], 'score': .8},
                   {'patient_id': 'n', 'center_mm': [0,0,0], 'score': .7}]
    result = evaluate(truth, predictions, distance_mm=2)
    assert (result['tp'], result['fp'], result['fn']) == (1, 2, 0)
    assert result['fp_per_case'] == 1 and result['negative_case_specificity'] == 0
    assert evaluate(truth, [])['fn'] == 1
    with pytest.raises(ValueError, match='outside'):
        evaluate(truth, [{'patient_id': 'foreign', 'center_mm': [0,0,0], 'score': .9}])


def test_split_patient_disjoint_and_reproducible():
    cases = [{'patient_id': f'p{i}', 'annotation_status': 'positive' if i < 15 else 'negative'} for i in range(30)]
    result = split_cases(cases)
    assert result == split_cases(cases[::-1])
    assert all(result.values())
    sets = [set(v) for v in result.values()]
    assert not sets[0] & sets[1] and not sets[1] & sets[2] and not sets[0] & sets[2]
    assert len(set.union(*sets)) == 30


def test_roi_never_expands_using_lvo(case, tmp_path):
    cow = np.zeros((24,24,24), dtype=np.uint8)
    cow[1,1,1] = 1
    nib.save(image(cow), case['cow'])
    with pytest.raises(ValueError, match='misses LVO'):
        export_dataset([case], {'train':['p001'], 'val':[], 'test':[]},
                       tmp_path / 'Task100_Miss', roi='cow-oracle', margin_mm=0)


def test_multiple_components_and_overlapping_proxies(case, tmp_path):
    lvo = np.zeros((24,24,24), dtype=np.uint8)
    lvo[10,10,10] = 1
    lvo[13,10,10] = 1
    nib.save(image(lvo), case['lvo'])
    assert len(targets(load_case(case))) == 2
    with pytest.raises(ValueError, match='overlapping'):
        export_dataset([case], {'train':['p001'], 'val':[], 'test':[]}, tmp_path / 'Task100_Overlap', box_mm=20)


def test_heldout_and_unknown_export(case, tmp_path):
    unknown = {**case, 'patient_id': 'unknown', 'annotation_status': 'unknown', 'lvo': None}
    output = tmp_path / 'Task100_Heldout'
    result = export_dataset([case, unknown], {'train':[], 'val':['unknown'], 'test':['p001']}, output)
    assert result['exported'] == 1 and result['excluded'] == 1
    assert (output / 'raw_splitted/imagesTs/p001_0000.nii.gz').is_file()
    assert not (output / 'raw_splitted/imagesTr').exists()


def test_pickle_requires_opt_in(tmp_path):
    with pytest.raises(ValueError, match='Pickle'):
        convert_predictions(tmp_path, tmp_path, tmp_path / 'result.json')


def test_reviewed_vessel_classification(case, tmp_path):
    task = tmp_path / 'Task101_Classify'
    reviews = {'p001': {'1': {'segment': 'M1', 'reviewed': True}}}
    export_dataset([case], {'train': ['p001'], 'val': [], 'test': []}, task, vessel_labels=reviews)
    props = json.loads((task / 'raw_splitted/labelsTr/p001.json').read_text())
    assert props['instances'] == {'1': 2}
    truth = audit_case(case)
    truth['targets'][0]['segment'] = 'M1'
    result = evaluate([truth], [{'patient_id': 'p001', 'center_mm': truth['targets'][0]['center_mm'],
                                'score': .9, 'segment': 'M1'}])
    assert result['classification']['accuracy_on_localized_targets'] == 1
    with pytest.raises(ValueError, match='Unreviewed'):
        export_dataset([case], {'train':['p001'], 'val':[], 'test':[]}, tmp_path / 'Task102_Unreviewed',
                       vessel_labels={'p001': {'1': {'segment':'BA', 'reviewed':False}}})


def test_inference_does_not_require_lvo_or_ground_truth_cow(case, tmp_path):
    from cta_lvo.inference import prepare_inference
    task = tmp_path / 'training' / 'Task100_LVO'
    export_dataset([case], {'train':['p001'], 'val':[], 'test':[]}, task)
    unlabeled = {'patient_id':'new', 'cta':case['cta'], 'space':'ncct'}
    output = tmp_path / 'inference' / task.name
    prepare_inference([unlabeled], task, output)
    assert (output / 'raw_splitted/imagesTs/new_0000.nii.gz').is_file()
    assert not (output / 'raw_splitted/labelsTs').exists()


def test_automatic_cow_required_at_inference(case, tmp_path):
    from cta_lvo.inference import prepare_inference
    task = tmp_path / 'training' / 'Task100_LVO'
    export_dataset([case], {'train':['p001'], 'val':[], 'test':[]}, task, roi='cow-oracle')
    with pytest.raises(ValueError, match='automatic'):
        prepare_inference([case], task, tmp_path / 'inference' / task.name, roi='cow-auto')


def test_scan_excludes_nested_environment_fixtures(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    pq.write_table(pa.table({'subject_id':['s1']}), root / 'train-00000.parquet')
    fixtures = root / 'venv' / 'fixtures'
    fixtures.mkdir(parents=True)
    pq.write_table(pa.table({'price':[10]}), fixtures / 'unrelated.parquet')
    assert inspect_parquet(root, tmp_path / 'schema.json')['shards'] == 1
    assert inspect_parquet(root, tmp_path / 'recursive.json', '**/*.parquet')['shards'] == 2


def test_isles_probe_actual_schema_and_misaligned_cta(case, tmp_path):
    from cta_lvo.data import probe_isles
    root = tmp_path / 'hf'
    root.mkdir()
    row = {'subject_id':'s1'}
    for field, key in [('ncct','cta'), ('cta','cta'), ('lvo_mask','lvo'), ('cow_segmentation','cow')]:
        row[field] = {'bytes': open(case[key], 'rb').read(), 'path': f'{field}.nii.gz'}
    cta = image(np.zeros((12,12,12), dtype=np.float32))
    row['cta']['bytes'] = gzip.compress(cta.to_bytes())
    pq.write_table(pa.Table.from_pylist([row]), root / 'train-00000.parquet')
    result = probe_isles(root, tmp_path / 'probe.json')[0]
    assert result['same_grid_as_ncct'] == {'cta':False, 'lvo_mask':True, 'cow_segmentation':True}
    assert result['images']['lvo_mask']['nonzero_voxels'] == 1
    assert result['subject_id'] == 's1'
