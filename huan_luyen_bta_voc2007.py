"""Prepare on CPU; explicitly train YOLOv8s+BTA on a matching GPU runtime.

Reuses the hash-locked, completed baseline engine and reviewed BTA geometry.
The baseline files, split lists and original JPEG/XML files are read-only.
Calling chuan_bi_bta() NEVER imports Torch, loads weights or starts training.
Calling huan_luyen_bta() is a separate, explicit action (10 epochs by default).
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
import csv
import hashlib
import io
import json
import math
import os
import sys

import numpy as np
from PIL import Image


SCRIPT_NAME = 'huan_luyen_bta_voc2007.py'
PROJECT = Path('/content/drive/MyDrive/Do_an_Manh_Chuong4')
VOC = Path('/content/voc2007_data/VOCdevkit/VOC2007')
RUN_NAME = 'yolov8s_bta_seed42'
BASELINE_NAME = 'yolov8s_baseline_seed42'
PREFLIGHT_NAME = 'bta_cpu_preflight_20261019_012305_148414_UTC'
ENGINE_NAME = 'huan_luyen_baseline_voc2007.py'
GEOMETRY_NAME = 'kiem_tra_bta_cpu_voc2007.py'
HELPER_NAME = 'huan_luyen_thu_yolov8s_voc2007.py'
ENGINE_SHA = '73f918134c275aefaca6c67a623f565fb83f598a53e826a22ee7721e5cc0b720'
GEOMETRY_SHA = 'ab69a9071318db00fb551af911bff58ca210915e685572ecbab08e9387661c8e'
HELPER_SHA = '7ef07079a86827c12e4a10fe44d655fc98a7c35778c585df28a67d88e3aa4523'
INITIAL_SHA = 'aebca1b9d2363d7e91e33b9675030f6a2865c1068bd441b8109bacd977c24939'
HISTORY_SHA = 'ad263ea2e23df1c8d2e729fc361db04c35b73e21160ff76873fa47b5fd198452'
TRAIN_SOURCE_SHA = '288f98fe7ffc8b8f2ec270c94e6d5de984549efae378d4691a302a36fac76c24'


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def save_new(path, content):
    with Path(path).open('xb') as stream:
        stream.write(content)


def load_module(name, content, filename):
    # Only our known/hash-checked sources are loaded; never an arbitrary ZIP entry.
    module = ModuleType(name)
    module.__file__ = str(filename)
    sys.modules[name] = module
    exec(compile(content, str(filename), 'exec'), module.__dict__)
    return module


def find_source(project, filename, expected_sha):
    baseline = project / 'formal_runs' / BASELINE_NAME
    bta = project / 'formal_runs' / RUN_NAME
    own = Path(__file__).parent if '__file__' in globals() else Path('/content')
    candidates = [bta / filename, baseline / filename,
                  project / 'logs' / PREFLIGHT_NAME / filename,
                  Path('/content') / filename, own / filename]
    candidates.extend(sorted(baseline.glob('invocation_*/' + filename)))
    candidates.extend(sorted(bta.glob('invocation_*/' + filename)))
    for path in candidates:
        if path.is_file():
            content = path.read_bytes()
            if sha256(content) == expected_sha:
                return content, path
    raise AssertionError(f'Thieu source da kiem tra: {filename}. '
                         'Khong chay lai baseline/pilot; gui thong bao nay de khoi phuc source.')


def adapt_engine_source(content):
    """Only rename run identity/status; numerical training code stays identical."""
    assert sha256(content) == ENGINE_SHA, 'Source baseline engine khong khop.'
    text = content.decode('utf-8')
    replacements = [
        ("SCRIPT_NAME = 'huan_luyen_baseline_voc2007.py'",
         "SCRIPT_NAME = 'huan_luyen_bta_voc2007.py'", 1),
        ("'yolov8s_baseline_seed42'", "'yolov8s_bta_seed42'", 1),
        ("'formal_baseline_resume'", "'formal_bta_resume'", 2),
        ("'formal_baseline_best_ema'", "'formal_bta_best_ema'", 2),
        ("'BASELINE_COMPLETE'", "'BTA_COMPLETE'", 1),
        ("'BASELINE_SEGMENT_COMPLETE'", "'BTA_SEGMENT_COMPLETE'", 1),
        ('THU MUC BASELINE:', 'THU MUC YOLOv8s+BTA:', 1),
        ('BAT DAU BASELINE TU KHOI TAO CHUNG; KHONG nap checkpoint pilot.',
         'BAT DAU BTA TU KHOI TAO CHUNG; KHONG nap baseline best.pt hay pilot.', 1),
        ('Baseline da du 100 epoch;', 'YOLOv8s+BTA da du 100 epoch;', 1),
        ('===== KET QUA BASELINE =====', '===== KET QUA YOLOv8s+BTA =====', 1),
    ]
    audit = []
    for before, after, count in replacements:
        assert text.count(before) == count, f'Engine patch khong khop: {before}'
        text = text.replace(before, after)
        audit.append({'before': before, 'after': after, 'occurrences': count})
    return text.encode('utf-8'), audit


def configure_engine(engine, geometry, helper, helper_path, project, voc, fingerprints):
    """Change train data only; keep val dataset/evaluator/optimizer/AMP/LR/EMA."""
    engine.PROJECT, engine.VOC = project, voc
    engine.RUN_DIR = project / 'formal_runs' / RUN_NAME
    baseline_dir = project / 'formal_runs' / BASELINE_NAME
    assert engine.RUN_DIR.resolve() != baseline_dir.resolve()
    original_config = engine.config_for
    original_dataset = engine.FormalDataset
    original_train_epoch = engine.train_epoch
    source_lookup = {row['image_id']: row for row in fingerprints}

    def config_for(metadata, pretrained_hash):
        config = original_config(metadata, pretrained_hash)
        config.update(protocol_version='voc07_original_bta_v1', method='YOLOv8s_BTA')
        config['train_augmentation']['BTA'] = True
        config['BTA_settings'] = deepcopy(geometry.SETTINGS)
        config['baseline_engine_sha256'] = ENGINE_SHA
        config['bta_geometry_sha256'] = GEOMETRY_SHA
        config['train_source_fingerprint_sha256'] = TRAIN_SOURCE_SHA
        config['baseline_reference_history_sha256'] = HISTORY_SHA
        return config

    class BTADataset(original_dataset):
        def __init__(self, loaded_helper, ids, mode):
            if mode == 'train':
                assert len(ids) == len(source_lookup) and set(ids) == set(source_lookup)
            super().__init__(loaded_helper, ids, mode)
            self._events = []
            self._seen = set()
            self._manifest_stream = None

        def begin_epoch(self, epoch, path):
            assert self.mode == 'train' and self._manifest_stream is None
            self.epoch, self._events, self._seen = epoch, [], set()
            self._manifest_stream = Path(path).open('x', encoding='utf-8')

        def __getitem__(self, index):
            if self.mode == 'val':
                # Exactly the original baseline validation path; never crop val.
                return super().__getitem__(index)
            assert self._manifest_stream is not None, 'Can mo manifest truoc khi hoc.'
            image_id, annotation = self.ids[index], self.annotations[index]
            assert image_id not in self._seen, 'Anh bi lay lap trong cung epoch.'
            updated, event = geometry.choose_crop(annotation, image_id, self.epoch)
            flip = engine.flip_for_image(image_id, self.epoch)
            assert flip == geometry.common_flip(image_id, self.epoch)
            with Image.open(voc / 'JPEGImages' / f'{image_id}.jpg') as source:
                assert source.size == (annotation['width'], annotation['height'])
                rgb = source.convert('RGB')
                image = rgb.crop(tuple(event['window'])) if event['applied'] else rgb
                row = geometry.letterbox_rgb(image, updated, flip, engine.IMGSZ)
            geometry.verify_row(row, updated, flip, engine.IMGSZ)
            record = {**event, **source_lookup[image_id], 'common_flip': flip}
            self._manifest_stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            self._events.append(event)
            self._seen.add(image_id)
            return row

        def finish_epoch(self, complete):
            if self._manifest_stream is not None:
                self._manifest_stream.flush()
                self._manifest_stream.close()
                self._manifest_stream = None
            if complete:
                assert len(self._events) == len(self.ids) and self._seen == set(self.ids)
                result = geometry.summarize_events(self._events)
                result.update(epoch=self.epoch + 1, manifest_complete=True)
                return result
            return None

    def train_epoch(model, ema, optimizer, scaler, dataset, loaded_helper, epoch, batch_csv):
        manifest = batch_csv.with_name(f'epoch_{epoch+1:03d}_bta_manifest.jsonl')
        dataset.begin_epoch(epoch, manifest)
        complete = False
        try:
            stats = original_train_epoch(model, ema, optimizer, scaler, dataset,
                                         loaded_helper, epoch, batch_csv)
            complete = True
        finally:
            bta_stats = dataset.finish_epoch(complete)
        engine.save_once(manifest.with_name(f'epoch_{epoch+1:03d}_bta_summary.json'),
                         json_bytes(bta_stats))
        stats['bta'] = bta_stats
        stats['bta_manifest_sha256'] = sha256(manifest.read_bytes())
        stats['bta_manifest'] = str(manifest)
        print(f"  BTA: yeu cau {bta_stats['requests']}; cat thanh cong {bta_stats['applied']}/"
              f"{bta_stats['images_examined']}; muc {bta_stats['applied_levels']}", flush=True)
        return stats

    engine.config_for, engine.FormalDataset, engine.train_epoch = config_for, BTADataset, train_epoch
    engine.load_helpers = lambda: (helper, helper_path)
    return original_config


def read_fingerprints(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4009 and len({row['image_id'] for row in rows}) == 4009
    assert all(set(row) == {'image_id', 'image_sha256', 'xml_sha256'} for row in rows)
    assert sha256(json_bytes(rows)) == TRAIN_SOURCE_SHA, 'Danh tinh anh TRAIN da thay doi.'
    return rows


def verify_train_sources(voc, fingerprints):
    for index, record in enumerate(fingerprints, 1):
        for subdir, extension, key in [('JPEGImages', 'jpg', 'image_sha256'),
                                        ('Annotations', 'xml', 'xml_sha256')]:
            path = voc / subdir / f"{record['image_id']}.{extension}"
            assert path.is_file(), 'Thieu VOC trainval trong runtime; can khoi phuc anh, khong chia lai tap.'
            assert sha256(path.read_bytes()) == record[key], f'Anh/nhan nguon da thay doi: {path}'
        if index % 1000 == 0 or index == len(fingerprints):
            print(f'Kiem tra anh/nhan TRAIN: {index}/{len(fingerprints)}', flush=True)


def assert_configs_differ_only_by_bta(baseline, proposed):
    restored = deepcopy(proposed)
    restored.update(protocol_version='voc07_original_baseline_v1', method='YOLOv8s_baseline')
    restored['train_augmentation']['BTA'] = False
    for key in ('BTA_settings', 'baseline_engine_sha256', 'bta_geometry_sha256',
                'train_source_fingerprint_sha256', 'baseline_reference_history_sha256'):
        restored.pop(key)
    assert restored == baseline, 'Cau hinh BTA khac baseline ngoai thay doi da khai bao.'


def load_context(project=None, voc=None):
    project = Path(project) if project is not None else PROJECT
    voc = Path(voc) if voc is not None else VOC
    assert (project / 'logs').is_dir(), 'Can ket noi Google Drive truoc.'
    baseline = project / 'formal_runs' / BASELINE_NAME
    run = project / 'formal_runs' / RUN_NAME
    sources, paths = {}, {}
    for name, expected in [(ENGINE_NAME, ENGINE_SHA), (GEOMETRY_NAME, GEOMETRY_SHA),
                            (HELPER_NAME, HELPER_SHA)]:
        sources[name], paths[name] = find_source(project, name, expected)
    geometry = load_module('bta_formal_checked_geometry', sources[GEOMETRY_NAME], paths[GEOMETRY_NAME])
    helper = load_module('bta_formal_checked_helper', sources[HELPER_NAME], paths[HELPER_NAME])
    adapted, patches = adapt_engine_source(sources[ENGINE_NAME])
    engine = load_module('bta_formal_checked_engine', adapted, paths[ENGINE_NAME])
    assert engine.INITIAL_SHA == INITIAL_SHA and engine.BATCH == 16 and engine.TOTAL_EPOCHS == 100
    assert engine.IMGSZ == 640 and engine.SEED == 42
    ids, metadata = helper.read_final_split(project)
    preflight_dir = project / 'logs' / PREFLIGHT_NAME
    preflight = json.loads((preflight_dir / 'summary.json').read_bytes())
    assert preflight['status'] == 'BTA_CPU_PREFLIGHT_PASSED'
    assert preflight['train_source_fingerprint_sha256'] == TRAIN_SOURCE_SHA
    fingerprints = read_fingerprints(preflight_dir / 'train_source_fingerprints.csv')
    assert [row['image_id'] for row in fingerprints] == ids['train']
    history_content = (baseline / 'history.csv').read_bytes()
    assert sha256(history_content) == HISTORY_SHA, 'History baseline khong trung ban 100 epoch da kiem tra.'
    history = list(csv.DictReader(io.StringIO(history_content.decode('utf-8-sig'))))
    assert [int(row['epoch']) for row in history] == list(range(1, 101))
    baseline_summary = json.loads((baseline / 'latest_summary.json').read_bytes())
    assert baseline_summary['status'] == 'BASELINE_COMPLETE'
    assert baseline_summary['completed_epochs'] == baseline_summary['planned_total_epochs'] == 100
    assert baseline_summary['initialization_sha256'] == INITIAL_SHA
    initial = project / 'checkpoints/initialization/yolov8s_voc20_seed42_init.pt'
    assert sha256(initial.read_bytes()) == INITIAL_SHA, 'Khoi tao chung khong khop; khong dung best.pt.'
    pretrained = project / 'checkpoints/pretrained/yolov8s.pt'
    pretrained_sha = sha256(pretrained.read_bytes())
    original_config = configure_engine(engine, geometry, helper, paths[HELPER_NAME],
                                      project, voc, fingerprints)
    baseline_config = json.loads((baseline / 'config.json').read_bytes())
    assert baseline_config == original_config(metadata, pretrained_sha)
    proposed_config = engine.config_for(metadata, pretrained_sha)
    assert_configs_differ_only_by_bta(baseline_config, proposed_config)
    environment_files = sorted(baseline.glob('invocation_*/environment.json'))
    assert environment_files, 'Thieu environment.json cua baseline; khong tu doan phien ban.'
    environments = [json.loads(path.read_bytes()) for path in environment_files]
    reference_keys = ('numpy', 'torch', 'torchvision', 'ultralytics', 'pillow',
                      'opencv_imported', 'torch_cuda', 'gpu')
    reference = {key: environments[0][key] for key in reference_keys}
    assert all(all(env[key] == reference[key] for key in reference_keys) for env in environments)
    snapshot = geometry.protected_baseline_snapshot(project)
    split_folder = project / 'splits/voc2007_seed42_final'
    split_hashes = {name: sha256((split_folder / name).read_bytes())
                    for name in ('train.txt', 'val.txt', 'test.txt', 'metadata.json')}
    verify_train_sources(voc, fingerprints)
    # Val JPEG/XML existence only; no test images/annotations are read.
    for image_id in ids['val']:
        assert (voc / 'JPEGImages' / f'{image_id}.jpg').is_file(), 'Thieu anh xac thuc goc.'
        assert (voc / 'Annotations' / f'{image_id}.xml').is_file(), 'Thieu nhan xac thuc goc.'
    geometry.geometry_selfcheck()
    parity_checks = 0
    for image_id in ids['train'][:12]:
        annotation = helper.read_annotation(voc / 'Annotations' / f'{image_id}.xml')
        for flip in (False, True):
            expected = helper.letterbox_and_labels(voc / 'JPEGImages' / f'{image_id}.jpg', annotation, flip)
            with Image.open(voc / 'JPEGImages' / f'{image_id}.jpg') as image:
                actual = geometry.letterbox_rgb(image.convert('RGB'), annotation, flip)
            for key in ('img', 'cls', 'bboxes'):
                np.testing.assert_array_equal(actual[key], expected[key])
            parity_checks += 1
    return {'project': project, 'voc': voc, 'run': run, 'engine': engine,
            'geometry': geometry, 'helper': helper, 'ids': ids, 'metadata': metadata,
            'fingerprints': fingerprints, 'sources': sources, 'adapted': adapted, 'patches': patches,
            'config': proposed_config, 'reference_environment': reference,
            'baseline_snapshot': snapshot, 'split_hashes': split_hashes, 'parity_checks': parity_checks}


def assert_protected_unchanged(context, check_sources=False):
    assert context['geometry'].protected_baseline_snapshot(context['project']) == context['baseline_snapshot']
    folder = context['project'] / 'splits/voc2007_seed42_final'
    assert {name: sha256((folder / name).read_bytes()) for name in context['split_hashes']} == context['split_hashes']
    if check_sources:
        verify_train_sources(context['voc'], context['fingerprints'])


def archive_protocol(context, folder, source_bytes):
    for name, content in context['sources'].items():
        context['engine'].save_once(folder / name, content)
    context['engine'].save_once(folder / 'bta_engine_adapted.py', context['adapted'])
    context['engine'].save_once(folder / 'engine_identity_patches.json', json_bytes(context['patches']))
    context['engine'].save_once(folder / SCRIPT_NAME, source_bytes)


def checked_source(source_bytes):
    if source_bytes is None:
        path = Path(__file__) if '__file__' in globals() else None
        assert path is not None and path.name == SCRIPT_NAME and path.is_file(), 'Can truyen source_bytes tu tep upload.'
        source_bytes = path.read_bytes()
    assert isinstance(source_bytes, bytes) and len(source_bytes) > 1000
    return source_bytes


def chuan_bi_bta(project=None, voc=None, source_bytes=None):
    """Safe CPU preparation. No training, package changes, or weight loading."""
    source_bytes = checked_source(source_bytes)
    context = load_context(project, voc)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC')
    output = context['project'] / 'logs' / f'bta_formal_preparation_{stamp}'
    output.mkdir(exist_ok=False)
    archive_protocol(context, output, source_bytes)
    save_new(output / 'proposed_config.json', json_bytes(context['config']))
    save_new(output / 'reference_gpu_environment.json', json_bytes(context['reference_environment']))
    assert_protected_unchanged(context)
    result = {'status': 'BTA_FORMAL_PREPARATION_PASSED', 'training_started': False,
              'weights_loaded': False, 'packages_installed': False,
              'formal_bta_training_completed': False, 'validation_evaluated': False, 'test_evaluated': False,
              'baseline_completed_epochs': 100, 'planned_bta_epochs': 100,
              'epochs_per_training_invocation': 10, 'physical_batch': 16, 'effective_batch': 16,
              'imgsz': 640, 'formal_training_device': 'cuda, matching baseline environment',
              'train_images': 4009, 'validation_images': 1002,
              'baseline_protected_files_unchanged': True, 'split_files_unchanged': True,
              'train_source_fingerprint_sha256': TRAIN_SOURCE_SHA,
              'initialization_sha256': INITIAL_SHA, 'baseline_best_weights_used': False,
              'baseline_preprocessing_parity_checks': context['parity_checks'],
              'baseline_controls_equal_except_declared_bta': True,
              'planned_run_directory': str(context['run']), 'output_directory': str(output),
              'script_sha256': sha256(source_bytes),
              'note': 'CPU preparation only. Explicit huan_luyen_bta() call and matching CUDA runtime required to train.'}
    save_new(output / 'summary.json', json_bytes(result))
    print('\n===== CHUAN BI BTA CHINH THUC =====', flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def check_gpu_environment(reference):
    import torch
    assert torch.cuda.is_available(), 'CHUA CO GPU: khong tu chuyen sang CPU de huan luyen chinh thuc.'
    import torchvision
    import ultralytics
    import PIL
    import cv2
    actual = {'numpy': np.__version__, 'torch': str(torch.__version__),
              'torchvision': str(torchvision.__version__), 'ultralytics': ultralytics.__version__,
              'pillow': PIL.__version__, 'opencv_imported': cv2.__version__,
              'torch_cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}
    differences = {key: {'baseline': value, 'current': actual[key]}
                   for key, value in reference.items() if actual[key] != value}
    assert not differences, ('Moi truong khac baseline; CHUA bat dau, KHONG tu cai lai Torch/CUDA. '
                             'Gui lai: ' + json.dumps(differences, ensure_ascii=False))
    return actual


def huan_luyen_bta(epochs_this_run=10, project=None, voc=None, source_bytes=None):
    """Explicit formal CUDA run; resume only this BTA run's complete epoch."""
    assert type(epochs_this_run) is int and 1 <= epochs_this_run <= 100
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    source_bytes = checked_source(source_bytes)
    context = load_context(project, voc)
    actual = check_gpu_environment(context['reference_environment'])
    engine = context['engine']
    engine.EPOCHS_THIS_RUN = epochs_this_run
    engine.uploaded = {SCRIPT_NAME: source_bytes}
    context['run'].mkdir(parents=True, exist_ok=True)
    archive_protocol(context, context['run'], source_bytes)
    print('BASELINE DA DU 100 EPOCH; GIU NGUYEN. CHI HUAN LUYEN YOLOv8s+BTA.', flush=True)
    print('GPU:', actual['gpu'], '| batch=16 | lich co dinh=100 epoch', flush=True)
    try:
        engine.main()
    finally:
        # Read-only checks; never repair or rewrite the protected baseline.
        assert_protected_unchanged(context, check_sources=True)
    path = context['run'] / 'latest_summary.json'
    return json.loads(path.read_bytes()) if path.is_file() else None


if __name__ == '__main__':
    # Deliberately safe when run as a script: prepare, NEVER auto-start training.
    chuan_bi_bta()
