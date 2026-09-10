#!/usr/bin/env bash
# Run after mouth crops and all requested model WAVs have passed validation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
PY="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
DATA="${ROOT_PREFIX}/zjw524/projects/data"
BENCH="${REPEAT213_DATASET:-$DATA/evaluations/celebvdub_train_repeat213_seed666_en_20260910}"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PATH="$(dirname "$PY"):$PATH"
"$PY" - "$BENCH" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);rows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()]
for r in rows:
    key=r['utterance_key'].removeprefix('celebvdub/')
    meta=root/'CelebVDub/video_mouth'/(key+'.json')
    m=json.loads(meta.read_text())
    assert not m['whole_frame_fallback'] and m['frames']==r['video_frames_25hz'] and m.get('detector_max_side')==640, key
assert len(list((root/'CelebVDub/video_mouth/train').rglob('*.mp4')))==213
print('Validated 213 mouth crops, original frame counts, no whole-face fallback')
PY
extract() {
    PYTHONPATH="src:$DATA/av_hubert/fairseq/fairseq" "$PY" -u src/aligndit/script/misc/extract_avhubert.py \
        --nshard 1 --rank 0 --v-input-dir "$BENCH/CelebVDub/video_mouth/train" --a-input-dir "$1/train" \
        --output-dir "$2/train" --ckpt-path "$DATA/large_vox_iter5.pt" --user_dir "$DATA/av_hubert/avhubert/avhubert"
}
# GT and generated audio use exactly the same newly prepared mouth videos.
validate_gt() {
"$PY" - "$BENCH" <<'PY'
import sys,json,numpy as np
from pathlib import Path
root=Path(sys.argv[1])
for r in map(json.loads,(root/'manifest.jsonl').read_text().splitlines()):
    key=r['utterance_key'].removeprefix('celebvdub/')
    feat=np.load(root/'CelebVDub/avhubert_feat'/(key+'.npy'))
    assert feat.shape==(r['video_frames_25hz'],1024) and np.isfinite(feat).all(), key
print('Validated 213 GT audio-visual features against manifest shapes')
PY
}
if ! validate_gt >/dev/null 2>&1; then extract "$BENCH/CelebVDub/audio" "$BENCH/CelebVDub/avhubert_feat"; fi
validate_gt
if [[ "${GT_ONLY:-0}" == 1 ]]; then exit 0; fi
for spec in ${EVAL_SPECS:-d1_tpca_150000 d1_tpca_200000 c2_svae_speaker_200000}; do
    out="$BENCH/results/$spec"
    verify() { "$PY" scripts/validate_tpca_eval.py "$out" "$1" --test-list "$BENCH/clips.lst" --split train --gt-feature-root "$BENCH/CelebVDub/avhubert_feat"; }
    verify wav
    if ! verify features >/dev/null 2>&1; then extract "$out" "$out/avhubert_feat"; fi
    verify features
    if ! verify avsync >/dev/null 2>&1; then
        "$PY" -u src/aligndit/script/eval/eval_celebvdub_test.py -e avsync -g "$out" -n 1 \
            --test-list "$BENCH/clips.lst" --dataset-root "$BENCH/CelebVDub" --split train --gt_av_feat "$BENCH/CelebVDub/avhubert_feat"
    fi
    verify all
    "$PY" scripts/score_repeat213.py "$out/_wer_results.jsonl"
    echo "ALL_METRICS_COMPLETE $spec $(date -Is)"
done
"$PY" scripts/score_repeat213.py "$BENCH/results/ground_truth/_wer_results.jsonl"
