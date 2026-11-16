"""One-level, hash-pinned provenance for physically imported LRS3 caches.

This module reads evidence only. Importers own copying assets and publishing
their per-ID receipts; the original cache and its locks are never modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


CONTRACT = 'reports/imported_source.json'
POLICY = 'lrs3_imported_source_v1'
IMPORT_POLICY = 'lrs3_sharded_import_v1'
COMPLETION = 'reports/sharded_import/completion.json'
LEDGER = 'reports/sharded_import/import_ledger.jsonl'
PLAN = 'reports/sharded_import/plan.json'
ARTIFACTS = {
    'source_manifest_sha256': 'manifests/source_inventory.jsonl',
    'source_contract_sha256': 'reports/source_contract.json',
    'media_quality_terminal_sha256': 'reports/media_full_pipeline/quality_terminal.json',
}
FIELDS = {'schema_version', 'policy', 'origin_root', 'destination_root', *ARTIFACTS}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def regular(root, relative):
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Imported artifact path escapes its physical root')
    path = root / relative
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError(f'Imported-source artifact must be a physical regular file: {path}')
    return path


def directory(value):
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError(f'Imported-source root must be an absolute physical directory: {value}')
    return path


def roots(origin, destination):
    origin, destination = directory(origin), directory(destination)
    if origin == destination or origin.is_relative_to(destination) or destination.is_relative_to(origin):
        raise ValueError('Imported-source roots must be disjoint physical directories')
    parent = origin / CONTRACT
    if parent.exists() or parent.is_symlink():
        raise ValueError('Nested or cyclic imported-source provenance is prohibited')
    return origin, destination


def build_import_contract(origin_root, destination_root):
    """Build a reviewed contract in memory; caller copies artifacts and writes it."""
    origin, destination = roots(origin_root, destination_root)
    from aligndit.lrs3_media_quality import load_terminal
    pins = {name: sha256(regular(origin, relative)) for name, relative in ARTIFACTS.items()}
    terminal = load_terminal(origin, expected_sha256=pins['media_quality_terminal_sha256'],
                             source_sha256=pins['source_manifest_sha256'])
    if terminal is None or terminal['source_contract_sha256'] != pins['source_contract_sha256']:
        raise ValueError('Imported source requires its complete original quality evidence chain')
    if any(sha256(regular(origin, relative)) != pins[name] for name, relative in ARTIFACTS.items()):
        raise ValueError('Original source changed while constructing the import contract')
    return {'schema_version': 1, 'policy': POLICY, 'origin_root': str(origin),
            'destination_root': str(destination), **pins}


def read_import_contract(root, expected_sha256=None):
    """Cheap contract/topology checks, NOT full source or physical-cache validation.

    Workers may use this after a fully validated coordinator pins the contract
    and their exact shard manifest. Each worker must still verify copied assets.
    """
    root = directory(Path(root).absolute())
    path = root / CONTRACT
    if not path.exists() and not path.is_symlink():
        if expected_sha256 is not None:
            raise ValueError('Pinned imported-source contract is missing')
        return None
    payload = regular(root, CONTRACT).read_bytes()
    fingerprint = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise ValueError('Imported-source contract differs from its pinned SHA256')
    contract = json.loads(payload)
    if (not isinstance(contract, dict) or set(contract) != FIELDS
            or type(contract.get('schema_version')) is not int or contract['schema_version'] != 1
            or contract.get('policy') != POLICY
            or any(not isinstance(contract.get(name), str) or not contract[name]
                   for name in ('origin_root', 'destination_root'))
            or any(not isinstance(contract.get(name), str) or not re.fullmatch('[0-9a-f]{64}', contract[name])
                   for name in ARTIFACTS)):
        raise ValueError('Malformed imported-source contract')
    origin, destination = roots(contract['origin_root'], contract['destination_root'])
    if destination != root:
        raise ValueError('Imported-source contract belongs to another destination')
    return {**contract, '_path': str(path), '_sha256': fingerprint}


def validate_imported_source(root, expected_sha256=None):
    """Validate original history at its original root and exact local copies.

    This never reads or acquires original writer locks and does not certify any
    new feature array. Physical-cache completion is independently audited later.
    """
    contract = read_import_contract(root, expected_sha256)
    if contract is None:
        return None
    origin, destination = Path(contract['origin_root']), Path(contract['destination_root'])
    for name, relative in ARTIFACTS.items():
        for owner in (origin, destination):
            if sha256(regular(owner, relative)) != contract[name]:
                raise ValueError(f'Imported-source original/local artifact changed: {owner / relative}')
    for relative in ARTIFACTS.values():
        old, new = (origin / relative).stat(), (destination / relative).stat()
        if (old.st_dev, old.st_ino) == (new.st_dev, new.st_ino):
            raise ValueError('Imported source artifacts must not share original writable inodes')
    from aligndit.lrs3_media_quality import load_terminal
    terminal = load_terminal(origin, expected_sha256=contract['media_quality_terminal_sha256'],
                             source_sha256=contract['source_manifest_sha256'])
    if terminal is None or terminal['source_contract_sha256'] != contract['source_contract_sha256']:
        raise ValueError('Imported-source original closure differs from its source contract')
    # Bind the destination independently, even though its bytes must match.
    source = json.loads(regular(destination, ARTIFACTS['source_contract_sha256']).read_bytes())
    if (source.get('source_inventory_sha256') != contract['source_manifest_sha256']
            or source.get('source_media_pairs') != terminal['requested']
            or source.get('original_clip_count') != terminal['original_ids']):
        raise ValueError('Imported-source destination has another canonical source scope')
    if read_import_contract(destination, contract['_sha256']) != contract:
        raise ValueError('Imported-source contract changed during validation')
    for name, relative in ARTIFACTS.items():
        for owner in (origin, destination):
            if sha256(regular(owner, relative)) != contract[name]:
                raise ValueError('Imported-source artifacts changed during validation')
    return {**contract, 'terminal': terminal}


def imported_source_scope(contract):
    if contract is None:
        return None
    return {'policy': POLICY, 'contract_path': CONTRACT, 'contract_sha256': contract['_sha256'],
            'origin_root': contract['origin_root'], 'destination_root': contract['destination_root'],
            **{name: contract[name] for name in ARTIFACTS}}


def imported_source_artifacts(contract):
    if contract is None:
        return []
    return [Path(contract['_path'])] + [Path(contract[owner]) / relative
        for owner in ('origin_root', 'destination_root') for relative in ARTIFACTS.values()]


def validate_import_completion(root, contract, canonical_rows=None):
    """Verify full merge coverage and its receipt ledger; not individual arrays."""
    root = directory(Path(root).absolute())
    if root != Path(contract['destination_root']):
        raise ValueError('Imported completion belongs to another cache root')
    completion_path, ledger_path, plan_path = [regular(root, p) for p in (COMPLETION, LEDGER, PLAN)]
    payload = completion_path.read_bytes()
    completion = json.loads(payload)
    plan = json.loads(plan_path.read_bytes())
    terminal = contract.get('terminal')
    if terminal is None:
        terminal = validate_imported_source(root, contract['_sha256'])['terminal']
    if (completion.get('schema_version') != 1 or completion.get('policy') != IMPORT_POLICY
            or completion.get('status') != 'complete'
            or completion.get('scope') != 'full'
            or completion.get('import_contract_sha256') != contract['_sha256']
            or completion.get('source_manifest_sha256') != contract['source_manifest_sha256']
            or completion.get('ledger_path') != LEDGER
            or sha256(ledger_path) != completion.get('ledger_sha256')
            or sha256(plan_path) != completion.get('plan_sha256')
            or any(type(completion.get(key)) is not int for key in
                   ('requested', 'processed', 'media_success', 'quality_excluded', 'failed'))
            or completion['requested'] != terminal['requested']
            or completion['processed'] != terminal['requested']
            or completion['media_success'] != terminal['verified']
            or completion['quality_excluded'] != terminal['excluded_count']
            or completion['failed'] != 0):
        raise ValueError('Imported cache requires a complete, exact full-source merge and pinned ledger/plan')
    if (plan.get('schema_version') != 1 or plan.get('policy') != IMPORT_POLICY
            or plan.get('scope') != 'full' or plan.get('requested') != completion['requested']
            or plan.get('quality_excluded') != completion['quality_excluded']
            or plan.get('origin_root') != contract['origin_root']
            or plan.get('destination_root') != contract['destination_root']
            or plan.get('import_contract_sha256') != contract['_sha256']
            or plan.get('source_manifest_sha256') != contract['source_manifest_sha256']):
        raise ValueError('Imported completion contradicts its full-source merge plan')
    def canonical():
        with regular(root, ARTIFACTS['source_manifest_sha256']).open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    identities = {}
    for row in canonical_rows if canonical_rows is not None else canonical():
        if row['sample_id'] in identities:
            raise ValueError('Imported canonical source has duplicate identities')
        identities[row['sample_id']] = row['relative_stem']
    receipts = {}
    with ledger_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            sid, stem = row.get('sample_id'), row.get('relative_stem')
            if (sid in receipts or sid not in identities or identities[sid] != stem
                    or row.get('import_receipt_path') != f'reports/import_receipts/{stem}.json'
                    or not isinstance(row.get('import_receipt_sha256'), str)
                    or not re.fullmatch('[0-9a-f]{64}', row['import_receipt_sha256'])):
                raise ValueError('Imported receipt ledger contains invalid, duplicate or foreign identities')
            receipts[sid] = row
    if len(identities) != terminal['requested'] or receipts.keys() != identities.keys():
        raise ValueError('Imported receipt ledger does not cover exactly the full canonical source')
    if (completion_path.read_bytes() != payload or sha256(ledger_path) != completion['ledger_sha256']
            or sha256(plan_path) != completion['plan_sha256']):
        raise ValueError('Imported completion/ledger/plan changed while validating')
    return {'scope': {'policy': IMPORT_POLICY, 'completion_path': COMPLETION,
                     'completion_sha256': hashlib.sha256(payload).hexdigest(),
                     **{key: completion[key] for key in ('scope', 'requested', 'processed', 'media_success',
                         'quality_excluded', 'failed', 'import_contract_sha256', 'source_manifest_sha256',
                         'plan_sha256', 'ledger_path', 'ledger_sha256')}},
            'artifacts': [completion_path, ledger_path, plan_path], 'receipts': receipts}


def validate_import_receipt(root, row, contract, ledger_entry, *, exclusion=None, local_hashes=None):
    """Check one new physical asset set against its pinned import receipt."""
    from aligndit.lrs3_media_quality import row_hash
    root, origin = directory(Path(root).absolute()), Path(contract['origin_root'])
    if root != Path(contract['destination_root']):
        raise ValueError('Imported receipt belongs to another cache root')
    if (ledger_entry.get('sample_id') != row['sample_id']
            or ledger_entry.get('relative_stem') != row['relative_stem']
            or ledger_entry.get('import_receipt_path') != f"reports/import_receipts/{row['relative_stem']}.json"):
        raise ValueError('Imported ledger entry does not bind this canonical identity')
    path = regular(root, ledger_entry['import_receipt_path'])
    payload = path.read_bytes()
    fingerprint = hashlib.sha256(payload).hexdigest()
    if fingerprint != ledger_entry['import_receipt_sha256']:
        raise ValueError('Imported per-ID receipt differs from the immutable ledger')
    receipt = json.loads(payload)
    if (receipt.get('schema_version') != 1 or receipt.get('policy') != IMPORT_POLICY
            or receipt.get('import_contract_sha256') != contract['_sha256']
            or receipt.get('source_record_sha256') != row_hash(row)
            or any(receipt.get(key) != row[key] for key in ('sample_id', 'relative_stem', 'original_clip_id'))
            or receipt.get('status') != ('quality_excluded' if exclusion else 'ok')
            or receipt.get('media_quality_reason') != (exclusion['reason'] if exclusion else None)):
        raise ValueError('Imported per-ID receipt differs from canonical source/quality provenance')
    stem = row['relative_stem']
    expected = {row['audio_relative_path']: row['audio_path'],
                row['latent_relative_path']: str(origin / row['latent_relative_path']),
                row['speaker_relative_path']: str(origin / row['speaker_relative_path']),
                f'state/audio_receipts/{stem}.json': str(origin / f'state/audio_receipts/{stem}.json')}
    if not exclusion:
        for key in ('mouth_relative_path', 'video_25hz_relative_path'):
            expected[row[key]] = str(origin / row[key])
        expected[f'reports/media_receipts/{stem}.json'] = str(origin / f'reports/media_receipts/{stem}.json')
        expected[f'state/video40_receipts/{stem}.json'] = None
        action = receipt.get('video40_action')
        if action not in {'copied', 'recomputed'}:
            raise ValueError('Imported 40-Hz cache has an unknown production action')
        expected[row['video_40hz_relative_path']] = str(origin / row[
            'video_40hz_relative_path' if action == 'copied' else 'video_25hz_relative_path'])
    files = receipt.get('files')
    if not isinstance(files, dict) or files.keys() != expected.keys():
        raise ValueError('Imported per-ID receipt omits or adds undeclared assets')
    hashes = local_hashes or {}
    for relative, source in expected.items():
        entry, local = files[relative], regular(root, relative)
        actual = hashes.get(relative) or sha256(local)
        if (entry.get('sha256') != actual or type(entry.get('size_bytes')) is not int
                or entry['size_bytes'] != local.stat().st_size or entry.get('source_path') != source):
            raise ValueError(f'Imported asset differs from its receipt: {relative}')
        if source is None:
            if entry.get('source_sha256') is not None:
                raise ValueError('Generated receipt may not claim a copied-source hash')
            continue
        source_path = Path(source)
        if sha256(source_path) != entry.get('source_sha256'):
            raise ValueError(f'Original imported asset changed: {source}')
        old, new = source_path.stat(), local.stat()
        if (old.st_dev, old.st_ino) == (new.st_dev, new.st_ino):
            raise ValueError('Imported assets must not share writable inodes with the original cache')
        recomputed = (not exclusion and relative == row['video_40hz_relative_path']
                      and receipt['video40_action'] == 'recomputed')
        if not recomputed and actual != entry['source_sha256']:
            raise ValueError('Copied imported asset differs from the original bytes')
    if path.read_bytes() != payload:
        raise ValueError('Imported per-ID receipt changed while validating')
    return fingerprint
