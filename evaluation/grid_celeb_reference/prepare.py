"""Freeze balanced targets/references before examining any generated outputs."""
import argparse
from collections import Counter, defaultdict
import json
import random

import numpy as np
import soundfile as sf

from protocol import DEFAULT_RUN, GRID, DATA, PROTOCOL, sha, save_json


def select_targets(rng):
    source = GRID / '0_Grid_Wav_22050_Abs_Feature/Grid_Wav_22050_Abs_Feature/val.txt'
    grouped = defaultdict(list)
    for line in source.read_text().splitlines():
        ident, speaker, phones, text = line.split('|')
        clip = ident.removeprefix(speaker + '-')
        words = [w for w in text.strip().lower().split() if w not in {'sil', 'sp', '<sil>'}]
        if len(words) != 6:
            raise ValueError('Not six words: ' + ident)
        lab = GRID / f'Grid_resample_ABS/Grid_Wav_22050_Abs/{speaker}/{ident}.lab'
        transcript = ' '.join(w for w in lab.read_text().lower().split() if w not in {'sil', 'sp', '<sil>'})
        if transcript != ' '.join(words):
            raise ValueError('Transcript disagreement: ' + ident)
        grouped[speaker].append((clip, ' '.join(words)))
    if set(grouped) != {f's{i}' for i in range(1, 35) if i != 21}:
        raise ValueError('Wrong GRID speaker coverage')
    selected = {}
    rejected = []
    for speaker in sorted(grouped, key=lambda s: int(s[1:])):
        pool = sorted(grouped[speaker])
        rng.shuffle(pool)
        # GRID full utterance videos have 75 frames at 25 Hz. This local copy
        # contains damaged files (2/4/9/63 video frames with ~3 s audio).
        # Filter solely on input completeness, before any model evaluation.
        complete = []
        for clip, text in pool:
            feature = GRID / f'Grid_lip_Feature/extrated_embedding_Grid_152_gray/{speaker}-face-{clip}.npy'
            frames = len(np.load(feature, mmap_mode='r', allow_pickle=False))
            if frames == 75:
                complete.append((clip, text))
            else:
                rejected.append({'target_id':f'test/{speaker}/{clip}', 'video_frames':frames,
                                 'reason':'incomplete GRID full-utterance video; expected 75 frames'})
        pool = complete
        if len(pool) < 20:
            raise ValueError('Insufficient complete GRID videos for '+speaker)
        counts = Counter()
        chosen = []
        # Random tie-breaking and greedy lexical coverage, without model scores.
        for _ in range(20):
            index = min(range(len(pool)), key=lambda j: sum(counts[(k, w)] for k, w in enumerate(pool[j][1].split())))
            item = pool.pop(index)
            chosen.append(item)
            counts.update(enumerate(item[1].split()))
        selected[speaker] = chosen
    return selected, source, rejected


def select_references(rng, run):
    candidates, rejected = defaultdict(list), []
    for ident in (DATA / 'celebvdub_test_s1.lst').read_text().splitlines():
        path = DATA / 'CelebVDub/audio/test' / (ident + '.wav')
        wav, sr = sf.read(path, dtype='float32', always_2d=True)
        mono = wav.mean(axis=1)
        duration = len(mono) / sr
        rms = float(np.sqrt(np.mean(mono ** 2)))
        clip_ratio = float(np.mean(np.abs(mono) >= .999))
        if not (sr == 16000 and 2 <= duration <= 5 and np.isfinite(mono).all() and rms > .001 and clip_ratio < .01):
            rejected.append({'id': ident, 'duration': duration, 'rms': rms, 'clipping_fraction': clip_ratio})
            continue
        text = (DATA / 'CelebVDub/text/test' / (ident + '.txt')).read_text().strip()
        if not text:
            raise ValueError('Empty reference text')
        candidates[ident.split('/')[0]].append((ident, path, text, len(mono)))
    groups = sorted(candidates)
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(candidates[group])
    chosen = []
    while len(chosen) < 66:
        before = len(chosen)
        for group in groups:
            if candidates[group] and len(chosen) < 66:
                chosen.append(candidates[group].pop())
        if len(chosen) == before:
            raise ValueError('Fewer than 66 quality-screened references')
    refs = []
    for ident, src, text, samples in chosen:
        dest = run / 'references/audio' / (ident + '.wav')
        dest.parent.mkdir(parents=True, exist_ok=True)
        wave, rate = sf.read(src, dtype='float32', always_2d=True)
        sf.write(dest, wave.mean(axis=1), rate, subtype='PCM_16')
        refs.append({'ref_id': 'celebvdub/' + ident, 'ref_speaker_id': ident.split('/')[0],
                     'ref_audio': str(dest), 'ref_audio_sha256': sha(dest), 'ref_source_audio': str(src.resolve()),
                     'ref_source_audio_sha256': sha(src), 'ref_text': text, 'ref_num_samples': samples,
                     'ref_frames': (samples + 399) // 400,
                     'ref_latent_path': str(run / 'references/latents' / (ident + '.npy')),
                     'ref_speaker_path': str(run / 'references/speakers' / (ident + '.npy'))})
    rng.shuffle(refs)
    return refs, rejected


def main(run):
    run = run.resolve()
    if (run / 'selection.json').exists():
        from protocol import rows
        rows(run)
        print('Existing immutable selection validated', flush=True)
        return
    if (run / 'inference.jsonl').exists():
        raise FileExistsError('Unrecorded selection; use another output directory')
    run.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    targets, source, rejected_targets = select_targets(rng)
    refs, rejected = select_references(rng, run)
    speakers = sorted(targets, key=lambda s: int(s[1:]))
    rng.shuffle(speakers)
    output = []
    for si, speaker in enumerate(speakers):
        for ti, (clip, text) in enumerate(targets[speaker]):
            # Balanced bipartite assignment: every reference appears for ten
            # different target speakers; each target speaker sees 20 distinct refs.
            ref = refs[(2 * si + 7 * ti) % 66]
            lip = GRID / f'Grid_lip_Feature/extrated_embedding_Grid_152_gray/{speaker}-face-{clip}.npy'
            face = GRID / f'Gird_FaceVAFeature/Grid_VA_feature/{speaker}-feature-{clip}.npy'
            a = np.load(lip, allow_pickle=False)
            b = np.load(face, allow_pickle=False)
            if a.ndim != 2 or a.shape[1] != 512 or b.shape != (len(a), 256) or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Invalid source feature shape: ' + speaker + '/' + clip)
            video = GRID / f'Grid_dataset_Raw/Grid_video/{speaker}/{clip}.mpg'
            if not video.is_file():
                raise FileNotFoundError(video)
            ident = f'test/{speaker}/{clip}'
            output.append({'protocol': PROTOCOL, 'pair_id': f'grid660/{speaker}/{clip}',
                           'target_id': ident, 'target_speaker_id': speaker, 'target_text': text,
                           'source_video': str(video), 'source_video_sha256': sha(video),
                           'native_frames': len(a), 'fps': 25, 'target_num_samples': len(a) * 640,
                           'target_frames': (len(a) * 640 + 399) // 400,
                           'source_lip_feature': str(lip), 'source_face_feature': str(face),
                           'mouth_video': str(run / 'assets/mouth' / (ident + '.mp4')),
                           'video_25hz_path': str(run / 'assets/video25' / (ident + '.npy')),
                           'target_video_path': str(run / 'assets/video40' / (ident + '.npy')),
                           'lip_feature': str(run / 'assets/lip' / (ident + '.npy')),
                           'face_feature': str(run / 'assets/face' / (ident + '.npy')),
                           **ref})
    # Interleave speakers so first 33 targets form the one-per-speaker smoke test.
    output = [output[si * 20 + ti] for ti in range(20) for si in range(33)]
    if set(Counter(x['ref_id'] for x in output).values()) != {10}:
        raise ValueError('Reference reuse is not balanced')
    (run / 'inference.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in output))
    scoring = [{**x, 'target_gt_audio': str(run / 'assets/gt' / (x['target_id'] + '.wav'))} for x in output]
    (run / 'pairs.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in scoring))
    save_json(run / 'selection.json', {'protocol': PROTOCOL, 'seed': 0, 'count': 660, 'reference_count': 66,
             'target_speaker_counts': dict(Counter(x['target_speaker_id'] for x in output)),
             'reference_counts': dict(Counter(x['ref_id'] for x in output)),
             'reference_source_video_groups': len({x['ref_speaker_id'] for x in output}),
             'reference_group_is_verified_person_identity': False,
             'reference_quality_screen': 'duration 2-5s, finite, RMS > -60dBFS, clipping fraction <1%; no human music/overlap audit',
             'reference_exclusions': rejected, 'grid_selection_source': str(source), 'grid_selection_source_sha256': sha(source),
             'input_completeness_exclusions': rejected_targets,
             'input_completeness_policy':'75 video frames, checked before generation; all 33 speakers retain 20 targets',
             'pairs_sha256': sha(run / 'pairs.jsonl'), 'inference_sha256': sha(run / 'inference.jsonl'),
             'duration_protocol': 'complete 25Hz video timeline; never use endpoint-trimmed GRID WAV as full-video GT',
             'target_audio_in_inference_manifest': False})
    print('Frozen 660 targets / 33 speakers / 66 references x 10', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=__import__('pathlib').Path, default=DEFAULT_RUN)
    main(p.parse_args().run)
