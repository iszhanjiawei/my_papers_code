#!/usr/bin/env bash
# Example: ROOT_PREFIX=/home JOINT_LAYERS=8 LIBRITTS_PRETRAINED_PATH=/path/model.pt bash this_script.sh
# Start persistent runs in a separate session with setsid and serve the reported
# TensorBoard logdir. This entry never falls back to the old S2c checkpoint.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/../../../.." && pwd)"
cd "$project_dir"
if [[ ! ${ROOT_PREFIX+x} ]]; then
  if [[ -x /home/zjw524/ENTER/envs/aligndit/bin/python ]]; then
    export ROOT_PREFIX=/home
  elif [[ -x /zjw524/ENTER/envs/aligndit/bin/python ]]; then
    export ROOT_PREFIX=""
  elif [[ -x /s7home/zjw524/ENTER/envs/aligndit/bin/python ]]; then
    export ROOT_PREFIX=/s7home
  else
    printf '%s\n' 'Set ROOT_PREFIX to the prefix containing zjw524/ENTER/envs/aligndit.' >&2
    exit 1
  fi
fi
source "$project_dir/env.sh"
python_bin="${PYTHON:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
joint_layers="${JOINT_LAYERS:-12}"
case "$joint_layers" in
  12) config_name=finetune_celebvdub_from_libritts_at ;;
  6|8) config_name="finetune_celebvdub_from_libritts_at_mm${joint_layers}" ;;
  *) printf '%s\n' 'JOINT_LAYERS must be 6, 8, or 12 (total depth stays 18).' >&2; exit 1 ;;
esac
if [[ ! -x "$python_bin" ]]; then
  printf 'Missing Python interpreter: %s\n' "$python_bin" >&2
  exit 1
fi
if [[ -z ${LIBRITTS_PRETRAINED_PATH:-} || ! -f "$LIBRITTS_PRETRAINED_PATH" ]]; then
  printf '%s\n' 'Set LIBRITTS_PRETRAINED_PATH to an existing LibriTTS text/audio checkpoint.' >&2
  exit 1
fi
export LIBRITTS_PRETRAINED_PATH
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
num_processes="${NUM_PROCESSES:-8}"
launch=("$python_bin" -u -m accelerate.commands.launch --num_processes "$num_processes"
        --num_machines 1 --mixed_precision bf16 --dynamo_backend no
        --main_process_port "${MAIN_PROCESS_PORT:-29562}")
if (( num_processes > 1 )); then
  launch+=(--multi_gpu)
fi
printf 'LibriTTS EMA -> CelebVDub: config=%s, parent=%s\n' "$config_name" "$LIBRITTS_PRETRAINED_PATH" >&2
exec "${launch[@]}" src/aligndit/script/train/finetune_semantic_vae_c2_direct_speaker.py \
  --config-name "$config_name" "$@"
