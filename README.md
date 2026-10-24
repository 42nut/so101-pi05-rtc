# SO101 · π0.5 Fine-tuning and RTC

SO101 黄色方块抓取放置项目：基础示范、纠错示范和策略运行后采集的 DAgger 数据混合训练，配套远程 RTC 推理及编译实验。

**状态：4000 步新一轮训练完成；该 checkpoint 真机评测待补。** 本仓库整理自实际运行文件，尚不是跨机器一键复现包。模型权重：[the42thnut/pi05-so101-cube-dagger](https://huggingface.co/the42thnut/pi05-so101-cube-dagger)（私有时需授权）。

## 已核实训练设置

| 项目 | 值 |
|---|---|
| 初始化 | recovery 阶段 `002000/pretrained_model` |
| 本轮更新 | 4000；不是加载完整优化器状态的 resume |
| 单卡 micro batch / 梯度累积 | 8 / 32，有效 batch 256 |
| 精度 | BF16，全量微调；gradient checkpointing 关闭 |
| 学习率 | 100 步线性 warmup，之后余弦衰减；峰值 1e-5，末值 1e-6 |
| 优化器 | AdamW，weight decay 0.01，梯度裁剪 1.0 |
| 数据 | 108 episodes、63791 frames、30 Hz |
| 划分 | 101 train / 7 held-out；59403 / 4388 frames |
| 有放回采样 | base / recovery / dagger = 30% / 30% / 40% |
| 本轮耗时 | 23:20:11（训练进度条记录） |
| 最后记录 loss | 0.0020313，仅训练 loss，不代表任务成功率 |

采样每个 update 有整数取整，五个 update 合计精确满足 3:3:4。保留集 episode 为 0、17、26、34、44、88、106；本包未提供其独立评测结果。配置名 `pi05_wsd` 是兼容标识，实际实现为余弦调度。日志的 `cumulative_update=8000` 来自脚本固定加 4000，不能仅据此验证此前累计训练步数。

## 文件导航

- `dagger_training_delivery/`：本轮训练器、启动器、原生训练源码参考、划分及 CPU 测试。
- `pi05_full_training/`：先前基础和 recovery 训练版本，保留为训练溯源。
- `data_preparation/`、`code/rebuild_dagger_30hz.py`：数据检查和重采样。
- `rtc_compiled_delivery/`：支持 eager / compiled 的服务端，默认 eager。
- `rtc_package/`：客户端、动作队列、mock 测试及早期服务端。
- 根目录 `*benchmark*.py`、`logs/*benchmark*.json`：性能实验代码与原始结果，多个历史版本未强行合并。
- `configs/final_checkpoint/`：最终 checkpoint 的四份 JSON；推理还需 HF 上的权重和 processor safetensors。
- `results/training_metrics.csv`、`training_summary.json`：从日志提取的 402 条记录（非逐步完整记录）。

原始子目录 README 保留作历史记录；当前状态和实验结论以本 README 与原始结果为准。

## 环境与复现边界

训练器明确要求 **LeRobot 0.6.1**，并核对 `native.train` 源码 SHA256：
`f2c2f33358951cede0ebc96604209c515d0daff26821867354feec81725bb4de`。
版本号相同也不保证源码一致；不要删除该检查。完整已验证的 pip/conda 环境导出及上游 Git commit 尚缺，不能将本文件当作锁定依赖清单。代码还使用 PyTorch、TorchCodec、Transformers、Accelerate、NumPy、Pillow 及 LeRobot 的机器人/相机依赖。

启动脚本保留原 AutoDL 的路径和动态库设置。换机器必须调整环境路径、缓存目录、数据目录和相机设备。数据及 recovery 初始化权重未包含在仓库中，训练前需自行准备。

在原训练服务器，先检查再运行（下方显式指定新输出目录，不覆盖已有训练）：

```bash
bash dagger_training_delivery/run_pi05_dagger.sh \
  --split "$PWD/dagger_training_delivery/reference_split.json" \
  --output-dir /root/autodl-tmp/pi05/outputs/pi05_dagger_reproduction \
  --until 4000 --check-only
```

通过后移除 `--check-only` 才会训练。默认 base 和 dataset 路径仍为原服务器路径；迁移时使用 `--base`、`--dataset-root` 并同步更新 split 中的 dataset_root。

## RTC 推理入口

原服务器上用最终权重运行 eager 服务端：

```bash
bash rtc_compiled_delivery/run_rtc_server.sh \
  --backend eager \
  --checkpoint /root/autodl-tmp/pi05/outputs/pi05_dagger_334_v2/checkpoints/004000/pretrained_model
```

服务端脚本默认 checkpoint 仍是旧 recovery 权重，因此必须显式传入上述路径。客户端见 `rtc_package/pi05_rtc_client.py`，它是服务器存档副本，尚未与本地机械臂实际使用版本比对。相机、校准 ID 和串口需按实际硬件配置。服务端无鉴权，远程访问应使用 SSH 隧道，不直接暴露公网端口。

## 编译实验：速度与一致性分别报告

`logs/rtc_compile_benchmark_20261023_135706.json` 在旧 recovery checkpoint 上记录：eager 中位数 322.98 ms，compiled 146.42 ms，约 2.21×。但同时 `parity_passed=false`、`integration_candidate=false`。不能据此声称通过一致性验证或已在最终 4000 步模型上取得同等加速。服务端保留显式实验开关，默认使用 eager。

## 已运行检查

```bash
python dagger_training_delivery/test_training_logic.py
python rtc_package/test_rtc.py
```

整理时 5 项训练逻辑测试、8 项 RTC/mock 测试通过；全部 Python 通过语法解析，Shell 通过 `bash -n`。未运行 GPU 模型加载、摄像头、机械臂或端到端真机验证。

## 发布前待补

1. 本地实际使用的 RTC 客户端与队列文件。
2. 训练环境导出及 LeRobot Git commit/本地改动。
3. 最终 checkpoint 真机评测记录和演示。
4. 数据集访问方式及最初模型来源。
5. 上游许可证核对与归属：本包未擅自为原生参考源码授予新许可证。

上游项目：[Hugging Face LeRobot](https://github.com/huggingface/lerobot)、[Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)。这是基于上游的应用和工程实验，非新的基础模型。
