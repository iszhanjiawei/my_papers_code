"""Symmetric GRID lexical normalization; no reference-text removal or ASR hints."""
import json
from pathlib import Path
import re

from jiwer import process_words

from protocol import rows, save_json, sha

NUMBERS = dict(zip('0123456789', 'zero one two three four five six seven eight nine'.split()))


def normalize(text):
    text = ' '.join(text.lower().split())
    text = re.sub(r'(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])', ' ', text)
    return re.sub(r'\b[0-9]\b', lambda m: NUMBERS[m.group()], text)


def report(run, method, gt=False):
    root = run/'assets/gt' if gt else run/'outputs'/method
    source = root/'_wer_results.jsonl'
    results = [json.loads(line) for line in source.read_text().splitlines()]
    expected = {row['target_id'] for row in rows(run)}
    if len(results) != 660 or {row['target_id'] for row in results} != expected:
        raise ValueError('WER coverage is not the frozen 660 targets')
    records = []
    edits = words = 0
    for row in results:
        truth, hypo = normalize(row['truth']), normalize(row['hypo'])
        alignment = process_words(truth, hypo)
        error = alignment.substitutions + alignment.deletions + alignment.insertions
        n = len(truth.split())
        if n != 6:
            raise ValueError('GRID target must retain all six words')
        edits += error
        words += n
        records.append({'target_id':row['target_id'], 'truth':truth, 'hypo':hypo,
            'substitutions':alignment.substitutions, 'deletions':alignment.deletions,
            'insertions':alignment.insertions, 'word_edit_distance':error, 'reference_word_count':n})
    summary = {'count':660, 'wer':edits/words, 'word_edit_distance':edits, 'reference_word_count':words,
        'aggregation':'corpus_word_edit_distance', 'asr_rerun':False,
        'normalization':'Symmetric lowercase/whitespace; split alphabet/digit boundaries; expand isolated single digits 0-9. No homophone corrections, no reference-text deletion.',
        'source_sha256':sha(source), 'manifest_sha256':sha(run/'pairs.jsonl')}
    save_json(root/'_wer_grid_normalized_summary.json', summary)
    (root/'_wer_grid_normalized_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return summary
