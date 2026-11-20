"""Frozen CelebV-Dub references and three WER protocols for this GRID snapshot."""
import hashlib
import json
import math
import re
import string
import wave
from pathlib import Path

from aligndit.script.eval.grid_s1_common import identity, read, sha

PROTOCOL = 'grid_svae_campplus_celebvdub_reference_target_text_only_v1'
PORTABLE_SHA256 = '9f533801de330c4a1911ffbc7c56147d6e9638a760fa61166478b18aafe4de3d'
WER_PROTOCOLS = {
    'original': 'legacy_lowercase_punctuation_no_digits_no_silence_corpus_v1',
    'digits_only': 'grid_digit_expansion_keep_silence_corpus_v1',
    'digits_silence': 'grid_digit_expansion_no_silence_corpus_v1',
}
TASKS = ('sim', 'wer', 'emosim', 'avsync')


def jsonl_bytes(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                              allow_nan=False)+'\n' for r in rows).encode()


def frozen_document(path):
    """Restore the document tables, independently of the other five models."""
    document = Path(path).read_text()

    def section(name):
        begin, end = f'<!-- ZERO_SHOT_{name}_BEGIN -->', f'<!-- ZERO_SHOT_{name}_END -->'
        if document.count(begin) != 1 or document.count(end) != 1:
            raise ValueError(f'Expected one frozen document section: {name}')
        return document.split(begin)[1].split(end)[0]

    def table(name, count):
        for line in section(name).splitlines():
            if not line.startswith('|'):
                continue
            fields = [v.strip().removeprefix('`').removesuffix('`') for v in line.strip().strip('|').split('|')]
            if fields[0] in ('编号', 'index') or re.fullmatch(r'[:\- ]+', fields[0]):
                continue
            if len(fields) != count:
                raise ValueError(f'Wrong table width: {name}')
            yield fields

    metadata = json.loads(re.search(r'```json\s*\n(.*?)\n```', section('METADATA'), re.S).group(1))
    references, labels = [], {}
    for label, source, text, wav_sha, txt_sha, rate, channels, frames, duration, rms, ref_id, raw_sha in table('REFERENCES', 12):
        if label in labels or ref_id != 'celebvdub/test/'+source:
            raise ValueError('Duplicate or invalid reference identity')
        ref = dict(id=ref_id, source_id=source, text=json.loads(text), audio_sha256=wav_sha,
                   text_sha256=txt_sha, raw_text_sha256=raw_sha, sample_rate=int(rate),
                   channels=int(channels), frames=int(frames), duration=float(duration), rms=float(rms),
                   audio_relative='audio/test/'+source+'.wav', text_relative='text/test/'+source+'.txt')
        labels[label] = ref
        references.append(ref)
    if len(references) != 33 or len({r['source_id'].split('/')[0] for r in references}) != 33:
        raise ValueError('Expected 33 different reference video groups')
    portable, assignments = [], {}
    for index, key, text, label in table('TARGETS', 4):
        if int(index) != len(portable) or not re.fullmatch(r's\d+/[a-z0-9]+', key):
            raise ValueError('Invalid target identity/order')
        ref = labels[label]
        speaker = key.split('/')[0]
        if assignments.setdefault(speaker, ref['id']) != ref['id']:
            raise ValueError('A target speaker has multiple external references')
        portable.append(dict(index=int(index), id=key, text=json.loads(text), reference_id=ref['id'],
                             reference_audio_relative=ref['audio_relative'], reference_text=ref['text'],
                             reference_audio_sha256=ref['audio_sha256']))
    blob = jsonl_bytes(portable)
    if len(portable) != 3281 or len({r['id'] for r in portable}) != 3281:
        raise ValueError('Invalid frozen coverage')
    if len(assignments) != 33 or len(set(assignments.values())) != 33:
        raise ValueError('Expected one-to-one speaker/reference assignments')
    if hashlib.sha256(blob).hexdigest() != PORTABLE_SHA256 or metadata['portable_pairs_sha256'] != PORTABLE_SHA256:
        raise ValueError('Frozen pairing/text/order SHA256 differs from the document')
    return portable, blob, references, assignments, metadata


def normalize_wer(text, variant):
    """Match zero-shot-Grid/evaluate_zeroshot.py, including legacy deletion."""
    if variant not in WER_PROTOCOLS:
        raise ValueError(f'Unknown WER variant: {variant}')
    if variant == 'original':
        from zhon.hanzi import punctuation
        for character in punctuation+string.punctuation:
            text = text.replace(character, '')
        text = text.replace('  ', ' ').lower()
    else:
        text = re.sub(r'[^\w\s]', '', text).lower()
        for digit, word in zip('0123456789', 'zero one two three four five six seven eight nine'.split()):
            text = text.replace(digit, ' '+word)
    return ' '.join(w for w in text.split() if variant != 'digits_silence' or w not in {'sp', 'sil'})


def edit_distance(truth, hypothesis):
    previous = list(range(len(hypothesis)+1))
    for i, word in enumerate(truth, 1):
        current = [i]
        for j, predicted in enumerate(hypothesis, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(word != predicted)))
        previous = current
    return previous[-1]


def wer_variants(raw_truth, raw_hypo):
    import jiwer
    result = {}
    for variant, protocol in WER_PROTOCOLS.items():
        truth, hypo = normalize_wer(raw_truth, variant), normalize_wer(raw_hypo, variant)
        if not truth:
            raise ValueError('Empty normalized reference')
        counts = jiwer.process_words(truth, hypo)
        errors = counts.substitutions+counts.deletions+counts.insertions
        if errors != edit_distance(truth.split(), hypo.split()):
            raise RuntimeError('Independent word edit distance disagrees')
        result[variant] = dict(truth=truth, hypo=hypo, hits=counts.hits, substitutions=counts.substitutions,
                               deletions=counts.deletions, insertions=counts.insertions,
                               word_edit_distance=errors, reference_word_count=len(truth.split()),
                               wer=errors/len(truth.split()), wer_protocol=protocol)
    return result


def summarize(results, task):
    if not results or len({r['id'] for r in results}) != len(results):
        raise ValueError('Empty or duplicate metric results')
    if task == 'wer':
        variants = {}
        for variant, protocol in WER_PROTOCOLS.items():
            items = [r['wer_variants'][variant] for r in results]
            counts = {k: sum(r[k] for r in items) for k in ('hits', 'substitutions', 'deletions', 'insertions')}
            errors = counts['substitutions']+counts['deletions']+counts['insertions']
            words = counts['hits']+counts['substitutions']+counts['deletions']
            if words <= 0:
                raise ValueError('Empty WER corpus denominator')
            variants[variant] = dict(count=len(results), **counts, word_edit_distance=errors,
                                     reference_word_count=words, wer=errors/words,
                                     display_value=f'{errors/words:.5f}', wer_protocol=protocol)
        return dict(count=len(results), wer=variants['digits_silence']['wer'], wer_variants=variants,
                    aggregation='corpus_word_edit_distance')
    keys = ('sim', 'sim_ref') if task == 'sim' else (task,)
    for r in results:
        if not all(math.isfinite(r[k]) for k in keys):
            raise ValueError('Nonfinite metric')
    return dict(count=len(results), aggregation='sample_mean',
                **{k: math.fsum(r[k] for r in results)/len(results) for k in keys})


def load_prepared(shared, smoke=False):
    shared = Path(shared)
    metadata = read(shared/'metadata.json')
    if metadata['status'] != 'complete' or metadata['count'] != 3274 or metadata['reference_count'] != 33:
        raise RuntimeError('Incomplete external-reference cache')
    if sha(shared/'manifest.jsonl') != metadata['manifest_sha256']:
        raise RuntimeError('External-reference manifest changed')
    rows = [json.loads(line) for line in (shared/'manifest.jsonl').read_text().splitlines()]
    if len(rows) != 3274 or len({identity(r) for r in rows}) != 3274:
        raise RuntimeError('Wrong target manifest coverage')
    if smoke:
        selected, seen = [], set()
        for row in rows:
            if row['reference_id'] not in seen:
                selected.append(row)
                seen.add(row['reference_id'])
        rows = selected
        if len(rows) != 33:
            raise RuntimeError('Smoke must cover every external reference')
    return rows, metadata


def check_output(root, row):
    root = Path(root)
    contract = read(root/'protocol.json')
    receipt = read(root/'receipts'/(identity(row)+'.json'))
    expected = dict(id=identity(row), reference_id=row['reference_id'], model_text=row['text'],
                    text_conditioning='target-only', protocol_sha256=sha(root/'protocol.json'),
                    reference_latent_sha256=row['reference']['ref_latent_sha256'],
                    reference_speaker_sha256=row['reference']['ref_speaker_sha256'],
                    prompt_frames=row['reference']['ref_frames'], target_frames=row['latent_frames'],
                    sampler_frames=row['reference']['ref_frames']+row['latent_frames'])
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f'Incorrect conditioning/geometry receipt: {identity(row)}')
    if contract['protocol'] != PROTOCOL or contract['text_conditioning'] != 'target-only':
        raise RuntimeError('Wrong inference protocol')
    if contract['reference_transcript_used_for_generation'] is not False:
        raise RuntimeError('Reference transcript was enabled')
    path = root/row['audio_relative_path']
    if receipt['sha256'] != sha(path):
        raise RuntimeError('Generated WAV changed')
    with wave.open(str(path)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) != (16000, 1, 2, row['original_num_samples']):
            raise RuntimeError('Wrong output audio format/duration')
    return receipt
