"""Independent GRID-target evaluation; never changes experimental snapshots."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

PREFIX = os.environ.get('ROOT_PREFIX', '')
PROJECT = Path(PREFIX + '/zjw524/projects/alignDiT_idea6')
DATA = Path(PREFIX + '/zjw524/projects/data')
GRID = Path(PREFIX + '/zjw524/datasets/Grid_Dataset')
REPA = PROJECT / 'my_papers_code/AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus'
BASELINE = PROJECT / 's8_baseline/alignDiT_baseline/AlignDiT'
STYLE = PROJECT / '对比实验/StyleDubber_Project_Compare/StyleDubber'
PRO = PROJECT / '对比实验/ProDubber_Project_Compare/ProDubber'
TOOLS = STYLE.parent / 'preprocess_celebvdub'
WEIGHTS = STYLE.parent / 'preprocess_models'
DEFAULT_RUN = DATA / 'evaluations/grid660_celeb66_seed0_v1'
PROTOCOL = 'grid660_celeb66_one_reference_per_target_v1'
COUNT = 660


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            d.update(chunk)
    return d.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rows(run, evaluation=False):
    path = Path(run) / ('pairs.jsonl' if evaluation else 'inference.jsonl')
    result = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(result) != COUNT or len({x['target_id'] for x in result}) != COUNT:
        raise ValueError('Expected 660 unique GRID targets')
    if any(x['protocol'] != PROTOCOL for x in result):
        raise ValueError('Wrong protocol')
    meta = json.loads((Path(run) / 'selection.json').read_text())
    if sha(path) != meta['pairs_sha256' if evaluation else 'inference_sha256']:
        raise ValueError('Frozen manifest changed')
    return result


def add_repa():
    sys.path.insert(0, str(REPA / 'src'))


def checkpoint(method):
    return {'repa': DATA / 'ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char/model_150000.pt',
            'baseline': PROJECT / 's8_baseline/model_100000.pt',
            'styledubber': PROJECT / '对比实验/celebV-Dub-checkpoints/styledubber/100000.pth.tar',
            'produbber': PROJECT / '对比实验/celebV-Dub-checkpoints/produbber/epoch_2nd_0012.pth'}[method]
