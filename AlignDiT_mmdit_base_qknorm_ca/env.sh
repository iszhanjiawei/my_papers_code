#!/usr/bin/env bash
# =====================================================================
# 路径前缀全局开关 (ROOT_PREFIX)
# ---------------------------------------------------------------------
# 本项目所有绝对路径都写成  ${ROOT_PREFIX}/zjw524/...
#   本机(当前服务器):  ROOT_PREFIX 留空  -> /zjw524/...
#   另一台服务器:      ROOT_PREFIX=/home -> /home/zjw524/...
#
# 切换方式(任选其一):
#   1) 改下面这一行的默认值, 例如另一台服务器改成:  "${ROOT_PREFIX:-/home}"
#   2) 或在 shell 里全局导出(推荐, 一处切换所有项目/脚本):
#        export ROOT_PREFIX=/home
#
# 说明:
#   - shell 脚本会自动 source 本文件;
#   - yaml 配置用 ${oc.env:ROOT_PREFIX,''} 读取(Hydra/OmegaConf);
#   - python 用 os.environ.get("ROOT_PREFIX", "") 读取.
# =====================================================================
export ROOT_PREFIX="${ROOT_PREFIX:-/home}"
