#!/usr/bin/env bash
set -euo pipefail

# Start through setsid. Pass the same Hydra overrides (especially model.name)
# used by train_tpca_4x4090.sh to inspect the matching run's event directory.
# TPCA_CONFIG_NAME is shared with the training launcher; TPCA_TB_PORT defaults
# to 6006. Check that the requested port is free before starting this service.
experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$experiment_root"
source "$experiment_root/env.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$experiment_root/src${PYTHONPATH:+:$PYTHONPATH}"

tpca_python="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
tpca_config="${TPCA_CONFIG_NAME:-finetune_celebvdub_mm_d1_hunyuan_tpca}"
tpca_logdir="$("$tpca_python" - "$experiment_root" "$tpca_config" "$@" <<'PY'
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

root = Path(sys.argv[1])
with initialize_config_dir(config_dir=str(root / "src/aligndit/config"), version_base="1.3"):
    cfg = compose(config_name=sys.argv[2], overrides=sys.argv[3:])
run_name = f"{cfg.model.name}_{cfg.model.mel_spec.mel_spec_type}_{cfg.model.tokenizer}_{cfg.datasets.name}"
print(root / "runs" / run_name)
PY
)"

printf 'TensorBoard logdir: %s\n' "$tpca_logdir"
printf 'TensorBoard host: %s; port: %s\n' "${TPCA_TB_HOST:-0.0.0.0}" "${TPCA_TB_PORT:-6006}"
if [[ "${TPCA_DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi
exec "$tpca_python" -u -m tensorboard.main \
    --logdir "$tpca_logdir" \
    --host "${TPCA_TB_HOST:-0.0.0.0}" \
    --port "${TPCA_TB_PORT:-6006}"
