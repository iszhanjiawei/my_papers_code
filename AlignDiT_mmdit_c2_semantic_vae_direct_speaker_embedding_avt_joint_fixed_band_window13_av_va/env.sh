#!/usr/bin/env bash
# =====================================================================
# 路径前缀全局开关 (ROOT_PREFIX)
# ---------------------------------------------------------------------
# 本项目所有绝对路径都写成  ${ROOT_PREFIX}/zjw524/...
#   ROOT_PREFIX 留空  -> /zjw524/...
#   ROOT_PREFIX=/home -> /home/zjw524/...
#
# 未设置时按已有 Python 环境自动检测；显式设置（包括空值）优先。
# 可在 shell 里全局导出，一处切换所有项目/脚本：
#        export ROOT_PREFIX=/home
#
# 说明:
#   - shell 脚本会自动 source 本文件;
#   - yaml 配置用 ${oc.env:ROOT_PREFIX,''} 读取(Hydra/OmegaConf);
#   - python 用 os.environ.get("ROOT_PREFIX", "") 读取.
# =====================================================================
if [[ -z "${ROOT_PREFIX+x}" ]]; then
    if [[ -x /zjw524/ENTER/envs/aligndit/bin/python ]]; then
        ROOT_PREFIX=""
    elif [[ -x /home/zjw524/ENTER/envs/aligndit/bin/python ]]; then
        ROOT_PREFIX="/home"
    else
        ROOT_PREFIX=""
    fi
fi
export ROOT_PREFIX
