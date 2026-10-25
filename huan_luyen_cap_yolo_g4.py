"""New G4 runs; reuses the old, hash-locked numerical training engine unchanged.

Explicit call only. Original T4 YOLO/Faster/SSD/splits are read-only.
"""
from copy import deepcopy
from pathlib import Path
import gc
import json
import math
import quy_trinh_g4_common as C

INITIAL_SHA = 'aebca1b9d2363d7e91e33b9675030f6a2865c1068bd441b8109bacd977c24939'


def dependencies():
    return (C.pinned('huan_luyen_thu_yolov8s_voc2007.py'),
            C.pinned('huan_luyen_baseline_voc2007.py'),
            C.pinned('kiem_tra_bta_cpu_voc2007.py'), C.pinned('huan_luyen_bta_voc2007.py'))


def pair_config(original, method, environment):
    config = deepcopy(original)
    # The same extra runtime policy is applied to both runs. Core math is unchanged.
    config['run_family'] = 'new_G4_pair_v1'
    config['actual_training_gpu_required'] = C.G4
    config['g4_pair_software'] = {k: environment[k] for k in
        ('numpy', 'pillow', 'torch', 'torchvision', 'torch_cuda', 'ultralytics', 'opencv_imported')}
    config['not_a_resume_of_T4_run'] = True
    config['g4_method'] = method
    config['new_G4_wrapper_sha256'] = C.sha_file(Path(__file__))
    config['new_G4_common_sha256'] = C.sha_file(C.ROOT/'quy_trinh_g4_common.py')
    return config


def normalize_pair(config):
    result = deepcopy(config)
    for k in ('g4_method', 'BTA_settings', 'baseline_engine_sha256', 'bta_geometry_sha256',
              'train_source_fingerprint_sha256', 'baseline_reference_history_sha256'):
        result.pop(k, None)
    result.update(protocol_version='voc07_original_baseline_v1', method='YOLOv8s_baseline')
    result['train_augmentation']['BTA'] = False
    return result


def audit_resume(torch, run, engine, config):
    path = run/'last.pt'
    if not path.exists():
        previous = run/'previous_last.pt'
        if previous.exists():
            path = previous
            print('last.pt chua commit; phuc hoi previous_last.pt, phat lai epoch dang do.', flush=True)
        else:
            assert not any((run/n).exists() for n in ('history.csv', 'best.pt')), \
                'Run dang do; KHONG tu hoc lai tu dau: ' + str(run)
            return 0
    state = torch.load(path, map_location='cpu', weights_only=True)
    n = state['completed_epochs']
    assert 1 <= n <= 100 and state['config_sha256'] == C.digest(C.jb(config))
    history = state['history']
    assert [r['epoch'] for r in history] == list(range(1, n+1))
    assert state['ema_updates'] == n * 251
    assert state['purpose'] == 'formal_baseline_resume'
    assert all(r['images_seen'] == 4009 and r['optimizer_updates'] == 251 for r in history)
    values = [r['map50_voc07'] for r in history]
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
    assert state['best_score'] == max(values) and state['best_epoch'] == values.index(max(values))+1
    # Original engine repairs a best file if its commit was interrupted after last.pt.
    if state['best_epoch'] == n:
        needs_repair = not (run/'best.pt').exists()
        if not needs_repair:
            best = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            needs_repair = not (best['config_sha256'] == state['config_sha256'] and
                               best['epoch'] == n and best['map50_voc07'] == state['best_score'])
            del best
        if needs_repair:
            engine.save_checkpoint('best.pt', dict(purpose='formal_baseline_best_ema',
                config_sha256=state['config_sha256'], epoch=n, map50_voc07=state['best_score'],
                classes=list(engine.CLASSES), model_state_dict=state['ema_state_dict'],
                validation_metrics=state['last_validation_metrics']))
    else:
        best = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        assert best['config_sha256'] == state['config_sha256']
        assert best['epoch'] == state['best_epoch'] and best['map50_voc07'] == state['best_score']
        del best
    if n == 100:
        engine.replace_owned(run/'history.csv', engine.history_csv(history))
    del state
    return n


def train_one(method, torch, env, voc, ids, metadata, epochs_this_run):
    helper, engine, geometry, bta = dependencies()
    run = C.PROJECT/'formal_runs'/C.RUNS[0 if method == 'baseline' else 1]
    engine.PROJECT, engine.VOC, engine.RUN_DIR = C.PROJECT, voc, run
    engine.EPOCHS_THIS_RUN = epochs_this_run
    engine.load_helpers = lambda: (helper, C.ROOT/'huan_luyen_thu_yolov8s_voc2007.py')
    helper.record_environment = lambda output: (C.once(output/'environment.json', C.jb(env)) or env)
    original_config = engine.config_for
    engine.config_for = lambda md, sha: pair_config(original_config(md, sha), method, env)
    fingerprints_path = C.PROJECT/'logs'/bta.PREFLIGHT_NAME/'train_source_fingerprints.csv'
    fingerprints = bta.read_fingerprints(fingerprints_path)
    assert [r['image_id'] for r in fingerprints] == ids['train']
    bta.verify_train_sources(voc, fingerprints)
    if method == 'bta':
        # The reusable wrapper assigns RUN_DIR itself: redirect its identities BEFORE
        # configuring it, otherwise it would target the original T4 BTA folder.
        bta.RUN_NAME, bta.BASELINE_NAME = C.RUNS[1], C.RUNS[0]
        bta.HISTORY_SHA = C.sha_file(C.PROJECT/'formal_runs'/C.RUNS[0]/'history.csv')
        bta.configure_engine(engine, geometry, helper, C.ROOT/'huan_luyen_thu_yolov8s_voc2007.py',
                             C.PROJECT, voc, fingerprints)
    assert engine.RUN_DIR == run and run.name not in C.OLD_YOLO
    config = engine.config_for(metadata, C.sha_file(C.PROJECT/'checkpoints/pretrained/yolov8s.pt'))
    if method == 'bta':
        other = json.loads((C.PROJECT/'formal_runs'/C.RUNS[0]/'config.json').read_bytes())
        assert normalize_pair(config) == normalize_pair(other), 'Hai run khac ngoai BTA.'
    run.mkdir(parents=True, exist_ok=True)
    count = audit_resume(torch, run, engine, config)
    C.once(run/'new_g4_policy.json', C.jb({'original_T4_runs_retained': True,
        'initialization_sha256': INITIAL_SHA, 'training_device': C.G4,
        'source_pins': C.PINS, 'method': method,
        'note': 'Same old engineering recipe, not a paper reproduction. No accuracy gain is assumed.'}))
    if count < 100:
        engine.main()
    state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
    result = dict(status='G4_YOLO_COMPLETE' if state['completed_epochs'] == 100 else 'G4_YOLO_SEGMENT_COMPLETE',
                  completed_epochs=state['completed_epochs'], planned_total_epochs=100,
                  best_epoch=state['best_epoch'], best_validation_map50_voc07=state['best_score'],
                  run_directory=str(run), gpu=C.G4, test_evaluated=False,
                  initialization_sha256=INITIAL_SHA)
    C.atomic(run/'g4_progress.json', C.jb(result))
    del state
    gc.collect()
    torch.cuda.empty_cache()
    print('\n===== TIEN DO YOLO G4 =====\n' + json.dumps(result, indent=2), flush=True)
    return result


def chay_cap_yolo_g4(epochs_this_run=100):
    assert isinstance(epochs_this_run, int) and 1 <= epochs_this_run <= 100
    assert (C.PROJECT/'logs').is_dir(), 'Mount dung Drive truoc.'
    torch, _, env = C.prepare_runtime()
    # Stops before any new model is allocated if SSD/Faster are unfinished.
    C.require_complete(torch, C.PROJECT, C.RUNS[2:])
    with C.job_lock():
        before = C.snapshot()
        try:
            helper, _, geometry, _ = dependencies()
            ids, metadata = helper.read_final_split(C.PROJECT)
            voc = C.restore_voc(helper, ('train', 'val'))['train']
            assert C.sha_file(C.PROJECT/'checkpoints/initialization/yolov8s_voc20_seed42_init.pt') == INITIAL_SHA
            geometry.geometry_selfcheck()
            # Prove both PIL pathways are bit-identical before training.
            from PIL import Image
            import numpy as np
            for image_id in ids['train'][:12]:
                ann = helper.read_annotation(voc/'Annotations'/f'{image_id}.xml')
                for flip in (False, True):
                    expected = helper.letterbox_and_labels(voc/'JPEGImages'/f'{image_id}.jpg', ann, flip)
                    with Image.open(voc/'JPEGImages'/f'{image_id}.jpg') as image:
                        actual = geometry.letterbox_rgb(image.convert('RGB'), ann, flip)
                    for key in ('img', 'cls', 'bboxes'):
                        np.testing.assert_array_equal(actual[key], expected[key])
            baseline = train_one('baseline', torch, env, voc, ids, metadata, epochs_this_run)
            if baseline['completed_epochs'] < 100:
                return {'baseline': baseline, 'bta': 'Chua bat dau; chay lai cung lenh.'}
            proposed = train_one('bta', torch, env, voc, ids, metadata, epochs_this_run)
            return {'baseline': baseline, 'bta': proposed}
        finally:
            assert C.snapshot() == before, 'Tep cu thay doi; dung de kiem tra.'
            gc.collect()
            torch.cuda.empty_cache()
