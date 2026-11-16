"""Physically import audited LRS3 assets into disjoint, resumable shards.

The origin is read-only. Every worker owns one shard and one non-inherited lock;
the coordinator never passes lock descriptors to children. A separate merger
validates the exact requested ID set before assembling a new standard layout.
No completed-source claim is made by a representative pilot.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from aligndit.lrs3_imported_source import (
    ARTIFACTS, CONTRACT, build_import_contract, read_import_contract,
    validate_imported_source,
)
from aligndit.lrs3_media_quality import PROTOCOL, excluded_map, row_hash


POLICY = 'lrs3_sharded_import_v1'
RUN = 'reports/sharded_import'
PLAN = RUN + '/plan.json'
LEDGER = RUN + '/import_ledger.jsonl'
COMPLETION = RUN + '/completion.json'
VIDEO_PROTOCOL = 'avhubert_video_25hz_to_40hz_linear_align_corners_false_v1'
CODEC_SHA = '7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516'
CAMPPLUS_SHA = '92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199'
STOP = []


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def payload(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()


def safe(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or any(x in ('..', '.') for x in relative.parts):
        raise ValueError(f'Unsafe relative path: {relative}')
    path = Path(root) / relative
    if path.resolve(strict=False) != path.absolute():
        raise ValueError(f'Output path has a symlink component: {path}')
    return path


def identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def snapshot_bytes(path, expected=None, *, regular=False):
    """One stable open-file snapshot, including replacement while reading."""
    path = Path(path)
    if regular and (path.is_symlink() or path.resolve(strict=True) != path.absolute()):
        raise ValueError(f'Expected physical source file: {path}')
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        data = handle.read()
        after = os.fstat(handle.fileno())
    if identity(before) != identity(after) or identity(after) != identity(path.stat()):
        raise ValueError(f'Source changed while reading: {path}')
    fingerprint = hashlib.sha256(data).hexdigest()
    if expected is not None and fingerprint != expected:
        raise ValueError(f'Source hash mismatch: {path}')
    return data, fingerprint


def atomic_bytes(path, data, *, identical_only=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f'Refusing symlink output: {path}')
    if path.exists():
        if digest(path) == hashlib.sha256(data).hexdigest():
            return
        if identical_only:
            raise ValueError(f'Immutable destination differs: {path}')
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}.{time.time_ns()}')
    try:
        with temporary.open('xb') as handle:
            handle.write(data)
            handle.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value, *, identical_only=False):
    atomic_bytes(path, payload(value), identical_only=identical_only)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


@contextmanager
def owned_lock(path):
    """Only this process owns the FD; never pass it to descendants."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handle = Path(path).open('a')
    acquired = False
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        os.set_inheritable(handle.fileno(), False)
        yield
    finally:
        if acquired:
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def install_signals():
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda number, frame: STOP.append(number))


def check_stop():
    if STOP:
        raise InterruptedError(f'Orderly stop requested by signal {STOP[0]}')


def check_array(data, shape, *, unit_norm=False):
    import numpy as np
    value = np.load(io.BytesIO(data), allow_pickle=False)
    if value.shape != tuple(shape) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f'Invalid float32 array: {value.shape}/{value.dtype}, expected {shape}')
    if unit_norm and abs(float(np.linalg.norm(value)) - 1.0) > 1e-4:
        raise ValueError('Speaker array is not unit norm')
    return value


def check_audio(data, row):
    import numpy as np
    import soundfile as sf
    value, sample_rate = sf.read(io.BytesIO(data), dtype='float32')
    if (sample_rate != 16000 or value.ndim != 1 or len(value) != row['original_num_samples']
            or not np.isfinite(value).all()):
        raise ValueError('Audio snapshot differs from canonical 16-kHz mono samples')


def check_mouth(path, row):
    import cv2
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_N_THREADS, 1])
    count = 0
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 25) > .001:
            raise ValueError('Copied mouth is not a readable 25-Hz video')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (96, 96):
                raise ValueError('Copied mouth dimensions differ from 96x96')
            count += 1
    finally:
        cap.release()
    if count != row['video_frames']:
        raise ValueError(f'Copied mouth frame count differs: {count}/{row["video_frames"]}')


def save_file(data_root, relative, data, *, source_path=None, source_sha256=None):
    path = safe(data_root, relative)
    atomic_bytes(path, data)
    fingerprint = hashlib.sha256(data).hexdigest()
    if path.is_symlink() or digest(path) != fingerprint:
        raise ValueError(f'New physical output differs after writing: {path}')
    if source_path is not None and path.exists() and Path(source_path).exists():
        a, b = path.stat(), Path(source_path).stat()
        if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
            raise ValueError('Imported output aliases the original inode')
    return {'sha256': fingerprint, 'size_bytes': len(data),
            'source_path': str(source_path) if source_path is not None else None,
            'source_sha256': source_sha256}


def copy_checked(data_root, relative, source, expected=None, *, shape=None,
                 unit_norm=False, audio_row=None, physical_source=False):
    data, fingerprint = snapshot_bytes(source, expected, regular=physical_source)
    if shape is not None:
        check_array(data, shape, unit_norm=unit_norm)
    if audio_row is not None:
        check_audio(data, audio_row)
    return save_file(data_root, relative, data, source_path=source, source_sha256=fingerprint)


def audio_identity(row, audio_hash):
    return {'audio_sha256': audio_hash, 'utterance_key': row['utterance_key'],
            'posterior_seed': row['posterior_seed'],
            'posterior_protocol': 'semantic_vae_posterior_sample_v1', 'codec_sha256': CODEC_SHA,
            'campplus_sha256': CAMPPLUS_SHA,
            'speaker_protocol': 'full_waveform_10s_repeat_fbank_mean_average_l2_v1'}


def video40_identity(row, source_hash):
    return {'source_sha256': source_hash, 'protocol': VIDEO_PROTOCOL,
            'utterance_key': row['utterance_key'], 'target_frames': row['latent_frames']}


def interpolate_video40(data, frames):
    """Exact existing worker operation order; no alternate interpolation."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    value = np.load(io.BytesIO(data), allow_pickle=False)
    tensor = torch.from_numpy(value).T.unsqueeze(0)
    aligned = F.interpolate(tensor, size=frames, mode='linear', align_corners=False)[0].T.contiguous().numpy()
    output = io.BytesIO()
    np.save(output, aligned, allow_pickle=False)
    return output.getvalue()


def process_row(row, proof, origin, data_root, contract_sha, exclusion=None):
    """Copy one complete, provenance-checked sample; publish its receipt last."""
    stem = row['relative_stem']
    files = {}
    audio_receipt_rel = f'state/audio_receipts/{stem}.json'
    audio_payload, audio_receipt_hash = snapshot_bytes(origin / audio_receipt_rel, regular=True)
    audio_receipt = json.loads(audio_payload)
    audio_data, audio_hash = snapshot_bytes(row['audio_path'])
    check_audio(audio_data, row)
    if audio_receipt.get('identity') != audio_identity(row, audio_hash):
        raise ValueError('Original audio receipt differs from source/protocol/seed')
    files[row['audio_relative_path']] = save_file(data_root, row['audio_relative_path'], audio_data,
        source_path=row['audio_path'], source_sha256=audio_hash)
    del audio_data
    for name, shape in [('latent', (row['latent_frames'], 64)), ('speaker', (192,))]:
        relative = row[name + '_relative_path']
        files[relative] = copy_checked(data_root, relative, origin / relative,
            audio_receipt['outputs'][name], shape=shape, unit_norm=name == 'speaker', physical_source=True)
    files[audio_receipt_rel] = save_file(data_root, audio_receipt_rel, audio_payload,
        source_path=origin / audio_receipt_rel, source_sha256=audio_receipt_hash)
    result = {'schema_version': 1, 'policy': POLICY, 'sample_id': row['sample_id'],
              'relative_stem': stem, 'original_clip_id': row['original_clip_id'],
              'source_record_sha256': row_hash(row), 'import_contract_sha256': contract_sha,
              'status': 'quality_excluded' if exclusion else 'ok',
              'media_quality_reason': exclusion['reason'] if exclusion else None,
              'video40_action': None, 'files': files}
    if any(proof.get(key) != row[key] for key in ('sample_id', 'relative_stem', 'original_clip_id', 'video_frames')):
        raise ValueError('Canonical sample differs from pinned independent media proof')
    if exclusion:
        if proof.get('status') != 'failed' or exclusion['relative_stem'] != stem:
            raise ValueError('Excluded sample differs from original proof')
    else:
        if proof.get('status') != 'ok':
            raise ValueError('Usable sample lacks an independent successful media proof')
        media_rel = f'reports/media_receipts/{stem}.json'
        media_payload, media_receipt_hash = snapshot_bytes(origin / media_rel, regular=True)
        media = json.loads(media_payload)
        source_video_hash = digest(row['video_path'])
        if (media.get('relative_stem') != stem or media.get('source_path') != row['video_path']
                or media.get('source_sha256') != source_video_hash
                or media.get('protocol_sha256') != PROTOCOL or media.get('features_complete') is not True
                or media.get('video_frames') != row['video_frames']
                or media.get('feature_sha256') != proof.get('feature_sha256')):
            raise ValueError('Original media receipt differs from source/protocol/independent proof')
        source25_rel = row['video_25hz_relative_path']
        data25, hash25 = snapshot_bytes(origin / source25_rel, media['feature_sha256'], regular=True)
        check_array(data25, (row['video_frames'], 1024))
        files[source25_rel] = save_file(data_root, source25_rel, data25,
            source_path=origin / source25_rel, source_sha256=hash25)
        mouth_rel = row['mouth_relative_path']
        files[mouth_rel] = copy_checked(data_root, mouth_rel, origin / mouth_rel)
        if row['width'] == 96 and files[mouth_rel]['sha256'] != source_video_hash:
            raise ValueError('Existing 96px mouth no longer equals its original video')
        check_mouth(safe(data_root, mouth_rel), row)
        files[media_rel] = save_file(data_root, media_rel, media_payload,
            source_path=origin / media_rel, source_sha256=media_receipt_hash)
        target_rel = row['video_40hz_relative_path']
        receipt40_rel = f'state/video40_receipts/{stem}.json'
        expected40 = video40_identity(row, hash25)
        try:
            old40_payload, _ = snapshot_bytes(origin / receipt40_rel, regular=True)
            old40_receipt = json.loads(old40_payload)
            if old40_receipt.get('identity') != expected40:
                raise ValueError('Original 40-Hz receipt identity differs')
            data40, hash40 = snapshot_bytes(origin / target_rel,
                old40_receipt['outputs']['video_40hz'], regular=True)
            check_array(data40, (row['latent_frames'], 1024))
            action, recompute_reason = 'copied', None
        except (OSError, ValueError, KeyError) as error:
            data40 = interpolate_video40(data25, row['latent_frames'])
            check_array(data40, (row['latent_frames'], 1024))
            hash40 = hashlib.sha256(data40).hexdigest()
            action, recompute_reason = 'recomputed', f'{type(error).__name__}: {error}'
        files[target_rel] = save_file(data_root, target_rel, data40,
            source_path=origin / (target_rel if action == 'copied' else source25_rel),
            source_sha256=hash40 if action == 'copied' else hash25)
        files[receipt40_rel] = save_file(data_root, receipt40_rel,
            payload({'identity': expected40, 'outputs': {'video_40hz': hash40}}))
        result.update(video40_action=action, video40_recompute_reason=recompute_reason)
    # Detect receipt replacement while its outputs were being copied.
    if digest(origin / audio_receipt_rel) != audio_receipt_hash:
        raise ValueError('Original audio receipt changed during copy')
    if not exclusion and digest(origin / media_rel) != media_receipt_hash:
        raise ValueError('Original media receipt changed during copy')
    receipt_rel = f'reports/import_receipts/{stem}.json'
    write_json(safe(data_root, receipt_rel), result)
    return result


def expected_files(row, excluded):
    stem = row['relative_stem']
    names = [row[k + '_relative_path'] for k in ('audio', 'latent', 'speaker')]
    names.append(f'state/audio_receipts/{stem}.json')
    if not excluded:
        names += [row[k + '_relative_path'] for k in ('mouth', 'video_25hz', 'video_40hz')]
        names += [f'reports/media_receipts/{stem}.json', f'state/video40_receipts/{stem}.json']
    return set(names)


def validate_local_receipt(row, result, data_root, contract_sha, excluded):
    if (result.get('schema_version') != 1 or result.get('policy') != POLICY
            or result.get('sample_id') != row['sample_id'] or result.get('relative_stem') != row['relative_stem']
            or result.get('original_clip_id') != row['original_clip_id']
            or result.get('source_record_sha256') != row_hash(row)
            or result.get('import_contract_sha256') != contract_sha
            or result.get('status') != ('quality_excluded' if excluded else 'ok')
            or result.get('media_quality_reason') != (excluded['reason'] if excluded else None)
            or set(result.get('files', {})) != expected_files(row, bool(excluded))):
        raise ValueError('Local import receipt identity/scope differs')
    for relative, info in result['files'].items():
        path = safe(data_root, relative)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != info['size_bytes']:
            raise ValueError(f'Incomplete local imported file: {relative}')
        if digest(path) != info['sha256']:
            raise ValueError(f'Changed local imported file: {relative}')


def shard_paths(root, index):
    directory = root / 'shards' / f'shard{index:04d}'
    return directory, directory / 'data', directory / 'complete.json'


def check_prior_guard(guard, plan_sha):
    """Called only while owning this shard lock; unknown/live guards fail closed."""
    if not guard.exists() and not guard.is_symlink():
        return
    if guard.is_symlink():
        raise ValueError('Shard writer guard must be a regular file')
    value = read_json(guard)
    if value.get('plan_sha256') != plan_sha or type(value.get('pid')) is not int or value['pid'] <= 0:
        raise ValueError('Existing shard guard belongs to an unknown writer/plan')
    if value['pid'] != os.getpid():
        try:
            state = Path(f'/proc/{value["pid"]}/stat').read_text().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            return
        if state != 'Z':
            raise RuntimeError('Prior shard writer PID is still alive; investigate before recovery')


def get_plan(root, expected=None):
    path = root / PLAN
    data, fingerprint = snapshot_bytes(path, expected, regular=True)
    plan = json.loads(data)
    if (plan.get('policy') != POLICY or plan.get('destination_root') != str(root)
            or plan.get('schema_version') != 1 or plan.get('scope') not in ('full', 'pilot')):
        raise ValueError('Import plan belongs to another destination/protocol')
    contract = read_import_contract(root, plan['import_contract_sha256'])
    if contract is None or contract['origin_root'] != plan['origin_root']:
        raise ValueError('Import plan differs from imported-source contract')
    if origin_guards(Path(plan['origin_root'])) != plan.get('origin_writer_guards'):
        raise ValueError('Original writer guard changed since import planning')
    return plan, fingerprint, contract


def origin_guards(origin):
    """Observe only; never acquire or alter the original writer locks/guards."""
    result = {}
    for feature in ('latents', 'speaker_embeddings', 'video_25hz', 'video_40hz'):
        relative = f'state/{feature}/WRITE_ACTIVE.json'
        path = origin / relative
        result[relative] = digest(path) if path.exists() else None
    return result


def selected_rows(root, selection):
    chosen = None
    if selection is not None:
        chosen_rows = read_rows(selection)
        chosen = {r['sample_id']: r for r in chosen_rows}
        if len(chosen) != len(chosen_rows) or not chosen:
            raise ValueError('Pilot selection is empty or contains duplicate sample IDs')
    result, seen, stems, keys = [], set(), set(), set()
    for row in read_rows(root / ARTIFACTS['source_manifest_sha256']):
        if row['sample_id'] in seen or row['relative_stem'] in stems or row['utterance_key'] in keys:
            raise ValueError('Duplicate canonical sample/stem/utterance key')
        seen.add(row['sample_id']); stems.add(row['relative_stem']); keys.add(row['utterance_key'])
        if chosen is None or row['sample_id'] in chosen:
            if chosen is not None and chosen[row['sample_id']] != row:
                raise ValueError('Pilot selection record differs from canonical source')
            result.append(row)
    if chosen is not None and not chosen.keys() <= seen:
        raise ValueError('Pilot contains noncanonical sample IDs')
    return result


def prepare(root, origin, num_shards, selection=None):
    if num_shards < 1:
        raise ValueError('At least one deterministic shard is required')
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True); origin = origin.resolve(strict=True)
    with owned_lock(root / RUN / 'prepare.lock'):
        if (root / PLAN).exists():
            raise FileExistsError('Plan is immutable; use run to resume an existing prepared root')
        contract = build_import_contract(origin, root)
        guard_pins = origin_guards(origin)
        for field, relative in ARTIFACTS.items():
            data, _ = snapshot_bytes(origin / relative, contract[field], regular=True)
            atomic_bytes(safe(root, relative), data, identical_only=True)
        write_json(root / CONTRACT, contract, identical_only=True)
        checked = validate_imported_source(root)
        rows = selected_rows(root, selection)
        proof_path = origin / 'reports/media_full_pipeline/independent_verification.jsonl'
        wanted = {r['sample_id'] for r in rows}
        proof = {r['sample_id']: r for r in read_rows(proof_path) if r['sample_id'] in wanted}
        if proof.keys() != wanted:
            raise ValueError('Selected canonical IDs lack exact independent proof coverage')
        exclusions = excluded_map(checked['terminal'])
        selected_exclusions = {key: value for key, value in exclusions.items() if key in wanted}
        exclusion_rel = RUN + '/exclusions.json'
        write_json(root / exclusion_rel, selected_exclusions, identical_only=True)
        shards = []
        for index in range(num_shards):
            records = rows[index::num_shards]
            manifest = RUN + f'/inputs/shard{index:04d}.jsonl'
            proof_rel = RUN + f'/inputs/shard{index:04d}.proof.jsonl'
            atomic_bytes(root / manifest, b''.join(payload(row) for row in records), identical_only=True)
            atomic_bytes(root / proof_rel, b''.join(payload(proof[row['sample_id']]) for row in records), identical_only=True)
            shards.append({'index': index, 'manifest': manifest, 'manifest_sha256': digest(root / manifest),
                           'proof': proof_rel, 'proof_sha256': digest(root / proof_rel), 'requested': len(records)})
        spec_path = origin / 'state/latents/spec.json'
        spec_data, spec_sha = snapshot_bytes(spec_path, regular=True)
        spec = json.loads(spec_data)
        spec.update(dataset_root=str(root), imported_from={'path': str(spec_path), 'sha256': spec_sha})
        write_json(root / 'state/latents/spec.json', spec, identical_only=True)
        plan = {'schema_version': 1, 'policy': POLICY, 'scope': 'pilot' if selection else 'full',
                'origin_root': str(origin), 'destination_root': str(root),
                'import_contract_sha256': checked['_sha256'],
                'source_manifest_sha256': checked['source_manifest_sha256'],
                'source_contract_sha256': checked['source_contract_sha256'],
                'media_quality_terminal_sha256': checked['media_quality_terminal_sha256'],
                'origin_writer_guards': guard_pins,
                'requested': len(rows), 'quality_excluded': len(selected_exclusions),
                'num_shards': num_shards, 'assignment': 'canonical_selected_order_index_mod_num_shards',
                'exclusions': exclusion_rel, 'exclusions_sha256': digest(root / exclusion_rel), 'shards': shards}
        validate_imported_source(root, checked['_sha256'])
        if origin_guards(origin) != guard_pins:
            raise ValueError('Original writer guard changed while preparing import')
        write_json(root / PLAN, plan, identical_only=True)
        print(json.dumps({'plan': str(root / PLAN), 'plan_sha256': digest(root / PLAN),
                          'scope': plan['scope'], 'requested': len(rows)}), flush=True)


def shard_inputs(root, plan, index):
    if not 0 <= index < plan['num_shards']:
        raise ValueError('Shard index outside fixed plan')
    shard = plan['shards'][index]
    for path, fingerprint in [(shard['manifest'], shard['manifest_sha256']),
                              (shard['proof'], shard['proof_sha256']),
                              (plan['exclusions'], plan['exclusions_sha256'])]:
        if digest(safe(root, path)) != fingerprint:
            raise ValueError('Pinned shard manifest/proof/exclusions changed')
    rows = read_rows(root / shard['manifest'])
    proof = {r['sample_id']: r for r in read_rows(root / shard['proof'])}
    if len(rows) != shard['requested'] or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Shard contains duplicate or missing canonical IDs')
    if proof.keys() != {r['sample_id'] for r in rows}:
        raise ValueError('Shard proof IDs differ from its manifest')
    return rows, proof, read_json(root / plan['exclusions'])


def worker(root, index, plan_sha, threads=1):
    import torch
    torch.set_num_threads(threads)
    import cv2
    cv2.setNumThreads(1)
    plan, fingerprint, contract = get_plan(root, plan_sha)
    rows, proofs, exclusions = shard_inputs(root, plan, index)
    directory, data_root, completion = shard_paths(root, index)
    with owned_lock(directory / 'worker.lock'):
        guard = directory / 'WRITE_ACTIVE.json'
        check_prior_guard(guard, fingerprint)
        if completion.exists():
            # An interrupted final guard removal is recoverable only while this
            # process owns the same shard lock and verifies every finished file.
            check_shard(root, plan, fingerprint, index, validate_files=True, allow_owned_stale_guard=True)
            guard.unlink(missing_ok=True)
            print(json.dumps({'shard': index, 'already_complete': True}), flush=True)
            return
        write_json(guard, {'pid': os.getpid(), 'sid': os.getsid(0), 'plan_sha256': fingerprint})
        records, failures = [], []
        progress = directory / f'attempt.{time.time_ns()}.jsonl'
        try:
            with progress.open('x', buffering=1) as log:
                for position, row in enumerate(rows):
                    check_stop()
                    receipt_rel = f'reports/import_receipts/{row["relative_stem"]}.json'
                    receipt_path = safe(data_root, receipt_rel)
                    try:
                        reused = False
                        if receipt_path.exists():
                            candidate = read_json(receipt_path)
                            try:
                                validate_local_receipt(row, candidate, data_root, contract['_sha256'],
                                                       exclusions.get(row['sample_id']))
                                result, reused = candidate, True
                            except (OSError, ValueError, KeyError):
                                pass
                        if not reused:
                            result = process_row(row, proofs[row['sample_id']], Path(plan['origin_root']),
                                data_root, contract['_sha256'], exclusions.get(row['sample_id']))
                        record = {'sample_id': row['sample_id'], 'relative_stem': row['relative_stem'],
                                  'status': result['status'], 'video40_action': result['video40_action'],
                                  'import_receipt_path': receipt_rel, 'import_receipt_sha256': digest(receipt_path)}
                        records.append(record)
                        log.write(json.dumps({**record, 'local_reused': reused}) + '\n')
                    except Exception as error:
                        item = {'sample_id': row['sample_id'], 'status': 'error',
                                'error': f'{type(error).__name__}: {error}'}
                        failures.append(item); log.write(json.dumps(item) + '\n')
                    if (position + 1) % 50 == 0:
                        write_json(directory / 'progress.json', {'pid': os.getpid(), 'processed': position + 1,
                            'requested': len(rows), 'failed': len(failures), 'heartbeat_unix': time.time()})
            get_plan(root, fingerprint)
            if failures:
                raise RuntimeError(f'{len(failures)} samples failed; see {progress}')
            records_path = directory / 'records.jsonl'
            atomic_bytes(records_path, b''.join(payload(row) for row in records), identical_only=True)
            write_json(completion, {'schema_version': 1, 'policy': POLICY, 'plan_sha256': fingerprint,
                'import_contract_sha256': contract['_sha256'], 'shard': index, 'requested': len(rows),
                'processed': len(records), 'failed': 0, 'records_sha256': digest(records_path),
                'quality_excluded': sum(r['status'] == 'quality_excluded' for r in records),
                'finished_unix': time.time()}, identical_only=True)
            guard.unlink()
        except BaseException as error:
            write_json(directory / f'failure.{time.time_ns()}.json',
                       {'pid': os.getpid(), 'error': f'{type(error).__name__}: {error}', 'progress': str(progress)})
            raise


def check_shard(root, plan, plan_sha, index, *, validate_files, allow_owned_stale_guard=False):
    directory, data_root, complete_path = shard_paths(root, index)
    if (directory / 'WRITE_ACTIVE.json').exists() and not allow_owned_stale_guard:
        raise RuntimeError(f'Shard still has an active/incomplete writer guard: {index}')
    complete = read_json(complete_path)
    records_path = directory / 'records.jsonl'
    if (complete.get('policy') != POLICY or complete.get('plan_sha256') != plan_sha
            or complete.get('shard') != index or complete.get('failed') != 0
            or complete.get('import_contract_sha256') != plan['import_contract_sha256']
            or complete.get('records_sha256') != digest(records_path)):
        raise ValueError('Shard completion differs from its fixed plan or records')
    rows, _, exclusions = shard_inputs(root, plan, index)
    records = read_rows(records_path)
    by_id = {r['sample_id']: r for r in records}
    if (len(by_id) != len(records) or by_id.keys() != {r['sample_id'] for r in rows}
            or complete.get('requested') != len(rows) or complete.get('processed') != len(rows)):
        raise ValueError('Shard completion is not its exact requested ID set')
    for row in rows:
        record = by_id[row['sample_id']]
        relative = f'reports/import_receipts/{row["relative_stem"]}.json'
        path = safe(data_root, relative)
        if record.get('import_receipt_path') != relative or digest(path) != record.get('import_receipt_sha256'):
            raise ValueError('Shard per-ID receipt differs from completion ledger')
        result = read_json(path)
        if record.get('status') != result.get('status') or record.get('relative_stem') != row['relative_stem']:
            raise ValueError('Shard record identity/status differs')
        if validate_files:
            validate_local_receipt(row, result, data_root, plan['import_contract_sha256'], exclusions.get(row['sample_id']))
    return rows, records


def merge_one(root, data_root, row, record, contract_sha, exclusion):
    receipt_path = safe(data_root, record['import_receipt_path'])
    if digest(receipt_path) != record['import_receipt_sha256']:
        raise ValueError('Shard import receipt changed before merge')
    receipt = read_json(receipt_path)
    validate_local_receipt(row, receipt, data_root, contract_sha, exclusion)
    files = {**receipt['files'], record['import_receipt_path']:
             {'sha256': record['import_receipt_sha256'], 'size_bytes': receipt_path.stat().st_size}}
    for relative, info in files.items():
        source, destination = safe(data_root, relative), safe(root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_symlink() or digest(destination) != info['sha256']:
                raise ValueError(f'Merge destination differs; refusing overwrite: {relative}')
        else:
            # Only new immutable shard inodes may be linked; never origin inodes.
            os.link(source, destination, follow_symlinks=False)
        if digest(destination) != info['sha256']:
            raise ValueError('Merged physical output failed its checksum')
    return record


def merge(root, plan_sha, threads=48):
    if threads < 1:
        raise ValueError('Merger thread count must be positive')
    plan, fingerprint, contract = get_plan(root, plan_sha)
    with owned_lock(root / RUN / 'merge.lock'):
        validate_imported_source(root, contract['_sha256'])
        if (root / COMPLETION).exists():
            raise FileExistsError('Merged import is immutable; inspect its completion receipt')
        canonical = {r['sample_id']: r for r in read_rows(root / ARTIFACTS['source_manifest_sha256'])}
        collected, seen = [], set()
        # All shards are proven complete before touching the standard layout.
        for index in range(plan['num_shards']):
            with owned_lock(shard_paths(root, index)[0] / 'worker.lock'):
                rows, records = check_shard(root, plan, fingerprint, index, validate_files=False)
                if seen.intersection(r['sample_id'] for r in rows):
                    raise ValueError('Overlapping shard sample IDs')
                for row in rows:
                    if canonical.get(row['sample_id']) != row:
                        raise ValueError('Shard row differs from canonical source')
                seen.update(r['sample_id'] for r in rows)
                collected.append((index, rows, records))
        if len(seen) != plan['requested'] or plan['scope'] == 'full' and seen != canonical.keys():
            raise ValueError('Merged shards do not cover the exact requested canonical source scope')
        guard = root / 'state/video_40hz/WRITE_ACTIVE.json'
        write_json(guard, {'pid': os.getpid(), 'stage': 'import_merge', 'plan_sha256': fingerprint})
        ledger, media_success, excluded, copied40, computed40 = [], 0, 0, 0, 0
        for index, rows, records in collected:
            directory, data_root, _ = shard_paths(root, index)
            with owned_lock(directory / 'worker.lock'):
                # No worker can alter a completed shard during this merge.
                check_shard(root, plan, fingerprint, index, validate_files=False)
                by_id = {r['sample_id']: r for r in rows}
                exclusions = read_json(root / plan['exclusions'])
                def merge_record(record):
                    check_stop()
                    return merge_one(root, data_root, by_id[record['sample_id']], record,
                                     contract['_sha256'], exclusions.get(record['sample_id']))
                with ThreadPoolExecutor(max_workers=threads) as pool:
                    for record in pool.map(merge_record, records):
                        ledger.append(record)
                        excluded += record['status'] == 'quality_excluded'
                        media_success += record['status'] == 'ok'
                        copied40 += record['video40_action'] == 'copied'
                        computed40 += record['video40_action'] == 'recomputed'
            write_json(root / RUN / 'merge_progress.json', {'processed': len(ledger),
                'requested': plan['requested'], 'last_shard': index, 'heartbeat_unix': time.time()})
        if excluded != plan['quality_excluded'] or media_success + excluded != plan['requested']:
            raise ValueError('Merged quality/accounting counts differ from the plan')
        validate_imported_source(root, contract['_sha256'])
        get_plan(root, fingerprint)
        ledger.sort(key=lambda r: r['sample_id'])
        atomic_bytes(root / LEDGER, b''.join(payload(record) for record in ledger), identical_only=True)
        completion = {'schema_version': 1, 'policy': POLICY, 'scope': plan['scope'],
            'status': 'complete', 'plan_sha256': fingerprint, 'import_contract_sha256': contract['_sha256'],
            'source_manifest_sha256': plan['source_manifest_sha256'],
            'requested': plan['requested'], 'processed': len(ledger), 'media_success': media_success,
            'quality_excluded': excluded, 'failed': 0, 'copied_video40': copied40, 'recomputed_video40': computed40,
            'ledger_path': LEDGER, 'ledger_sha256': digest(root / LEDGER),
            'records_sha256': digest(root / LEDGER), 'total': len(ledger), 'finished_unix': time.time()}
        guard.unlink()
        write_json(root / COMPLETION, completion, identical_only=True)
        print(json.dumps(completion), flush=True)


def run(root, plan_sha, workers, do_merge, merge_threads=48):
    if workers < 1:
        raise ValueError('Worker count must be positive')
    plan, fingerprint, contract = get_plan(root, plan_sha)
    with owned_lock(root / RUN / 'coordinator.lock'):
        validate_imported_source(root, contract['_sha256'])
        state_path = root / RUN / 'state.json'
        state = {'pid': os.getpid(), 'sid': os.getsid(0), 'plan_sha256': fingerprint,
                 'status': 'running', 'workers': workers, 'finished': [], 'active': []}
        active, failed = {}, []
        todo = list(range(plan['num_shards']))
        try:
            while todo or active:
                check_stop()
                while todo and len(active) < workers:
                    index = todo.pop(0)
                    log_path = root / RUN / f'shard{index:04d}.{time.time_ns()}.log'
                    log = log_path.open('x')
                    command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--root', str(root),
                               '--plan-sha256', fingerprint, '--shard', str(index), '--threads', '1']
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True, close_fds=True)
                    active[index] = (process, log, str(log_path))
                for index, (process, log, log_path) in list(active.items()):
                    code = process.poll()
                    if code is not None:
                        process.wait(); log.close(); active.pop(index)
                        state['finished'].append({'shard': index, 'pid': process.pid,
                                                  'returncode': code, 'log': log_path})
                        if code != 0:
                            failed.append(index)
                state.update(active=[{'shard': i, 'pid': v[0].pid, 'log': v[2]} for i, v in active.items()],
                             heartbeat_unix=time.time())
                write_json(state_path, state)
                if active:
                    time.sleep(1)
            if failed:
                raise RuntimeError(f'Failed shards {failed}; verified previous shard files remain resumable')
            validate_imported_source(root, contract['_sha256'])
            state.update(status='shards_complete', active=[], finished_unix=time.time())
            write_json(state_path, state)
            if do_merge:
                merge(root, fingerprint, merge_threads)
                state.update(status='merged', finished_unix=time.time())
                write_json(state_path, state)
        except BaseException as error:
            # Parent termination is coordinated; owners release only their own locks.
            for process, _, _ in active.values():
                if process.poll() is None:
                    process.terminate()
            for process, log, _ in active.values():
                process.wait(); log.close()
            state.update(status='stopped_or_failed', error=f'{type(error).__name__}: {error}',
                         active=[], finished_unix=time.time())
            write_json(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'worker', 'merge'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--origin', type=Path)
    parser.add_argument('--selection', type=Path, help='Explicit representative pilot; never full completion')
    parser.add_argument('--shards', type=int, default=64)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--shard', type=int)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--merge-workers', type=int, default=48)
    args = parser.parse_args()
    install_signals()
    root = args.root.absolute()
    if root.is_symlink() or root.resolve(strict=False) != root:
        raise ValueError('Destination root must be a physical absolute directory')
    if args.mode == 'prepare':
        if args.origin is None:
            parser.error('prepare requires --origin')
        prepare(root, args.origin.absolute(), args.shards, args.selection)
    else:
        if not args.plan_sha256:
            parser.error('The exact prepared --plan-sha256 is required')
        if args.mode == 'run':
            run(root, args.plan_sha256, args.workers, args.merge, args.merge_workers)
        elif args.mode == 'worker':
            if args.shard is None or args.threads < 1:
                parser.error('worker requires --shard and positive --threads')
            worker(root, args.shard, args.plan_sha256, args.threads)
        else:
            merge(root, args.plan_sha256, args.merge_workers)


if __name__ == '__main__':
    main()
