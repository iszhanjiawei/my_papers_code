#!/usr/bin/env bash
# S1 uses a same-clip prompt. S2 uses the GT dialogue dubbing manifest and an independent prompt.
set -euo pipefail
setting="${1:-}"
case "$setting" in
    s1|s2) shift ;;
    *) echo "Usage: bash infer_celebvdub_concat_text.sh {s1|s2} [inference arguments]" >&2; exit 2 ;;
esac
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
export PYTHONPATH="$project_root/src"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
config="$project_root/src/aligndit/config/finetune_celebvdub_concat_text.yaml"
save_dir="$("$python_bin" - "$config" <<'PY'
from pathlib import Path
import sys
from hydra import compose, initialize_config_dir
config = Path(sys.argv[1])
with initialize_config_dir(version_base="1.3", config_dir=str(config.parent)):
    print(compose(config_name=config.stem).ckpts.save_dir)
PY
)"
step="${CKPT_STEP:-200000}"
checkpoint="${CHECKPOINT:-$save_dir/model_${step}.pt}"
output_dir="${OUTPUT_DIR:-$save_dir/eval_${setting}_${step}_concat_text}"
if [[ "$setting" == s1 ]]; then
    export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
    exec "$python_bin" -u -m aligndit.script.eval.infer_celebvdub_semantic_vae_s1 \
        --config "$config" --checkpoint "$checkpoint" --step "$step" --output-dir "$output_dir" "$@"
fi
exec "$python_bin" -u -m aligndit.script.eval.infer_celebvdub_setting2_dubbing \
    --config "$config" --checkpoint "$checkpoint" --step "$step" --output-dir "$output_dir" \
    --reference-cache "${REFERENCE_CACHE:-$save_dir/eval_s2_reference_cache}" \
    --gpu "${EVAL_GPU:-0}" "$@"
