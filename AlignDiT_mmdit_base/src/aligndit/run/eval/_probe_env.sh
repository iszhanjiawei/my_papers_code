#!/bin/bash
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
printf "found=%s PYTHON=%s/zjw524/ENTER/envs/aligndit/bin/python\n" "$__envdir" "${ROOT_PREFIX}"
