# GRID 独立实验

本目录的目标数据集是 GRID；这是用户指定的 CelebV-Dub 模型的独立副本。
本目录以 `README.md`、`finetune_grid.yaml` 和 GRID 专用入口为准。
父目录关于源码隔离、Git 跟踪、后台任务、环境及 TensorBoard 的要求继续适用。

- 原模型目录不得随本实验自动修改。所有命令显式使用本目录 `PYTHONPATH=src`。
- 用户已确认沿用 GRID 数据目录现有 train.txt / val.txt；排除项必须保留原因。
- 原 WAV 存在静音裁剪；使用原始 MPG 完整音轨并与嘴部帧时间轴对齐。
- 模型需要 AV-HuBERT 1024 维视频特征，不能直接使用数据集的 512 维唇部缓存。
- 下游沿用原 GRID 字符词表和 LibriSpeech latent 标准化，固定特征模型哈希。LibriTTS 50k EMA 按字符身份迁移。
- `--limit` 缓存仅作检查；正式训练须读取全量审计通过的缓存，val 不进入训练。
- 特征、数据清单、审计输出、日志和 checkpoint 是运行产物，存于仓库外或忽略目录。
- 本副本继承的 CelebV-Dub / LRS3 入口不代表 GRID 训练或评测协议；GRID 训练使用
  `src/aligndit/run/train/finetune_grid.sh`。
