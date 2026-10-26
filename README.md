# SO-ARM101 × π₀.₅：全参数微调与真机闭环部署

基于 LeRobot 的 π₀.₅ VLA 模型，在 SO-ARM101 主从机械臂平台上完成黄色方块抓取放置任务。项目覆盖双视角示范采集、基础/纠错/策略运行后补充数据的分阶段训练，以及云端推理与本地动作执行的 RTC 集成。

**项目进展：已完成三阶段训练与真机测试，50 次抓取放置测试成功 47 次（94%）。** 测试中改变方块位置，并观察到策略自主纠错。该结果是作者记录的当前桌面单任务结果，不代表跨物体、跨场景泛化能力；逐次评测记录与演示视频仍待补齐。

模型权重：[the42thnut/pi05-so101-cube-dagger](https://huggingface.co/the42thnut/pi05-so101-cube-dagger)（私有时需授权）。仓库整理自实际运行文件，尚不是跨机器一键复现包。

## 项目工作与实现边界

- **真机数据闭环**：使用主从臂遥操作采集基础示范，针对抓取偏差、空抓等失败状态补充纠错示范，再结合策略运行暴露的问题采集 DAgger 阶段数据。
- **全参数微调**：在单张 A800 80GB 上训练，不冻结视觉编码器或其他模型部分，不使用 LoRA；采用 BF16 和梯度累积。本轮实际配置关闭 gradient checkpointing。
- **远程部署与诊断**：集成已有 RTC 实现与动作队列机制，连接云端动作预测和本地执行，处理推理期间已消耗动作的裁剪、过期响应、队列耗尽和静止重建，并记录请求延迟、关节跟踪误差与动作切换差异。
- **评测与实验**：完成 50 次真机任务测试，保留训练日志和编译性能实验；RTC 算法及队列机制不作为本项目原创算法贡献。

## 任务与模型接口

| 项目 | 设置 |
|---|---|
| 任务 | 将黑色框选区域内的黄色方块抓取并放入灰蓝色托盘 |
| 观测 | 第三视角 front + 腕部 wrist RGB、语言指令、6 维关节状态 |
| 原始图像 | 两路 640 × 480，30 FPS；模型配置图像分辨率 224 × 224 |
| 状态/动作维度 | 6：shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper |
| 动作形式 | 绝对关节位置目标；6 维包含夹爪，不表示 6 个空间关节自由度 |
| 动作块 | 50 × 6；配置 Flow 推理步数为 10 |
| 本地执行 | 目标 30 Hz；不等于模型每秒完成 30 次推理 |

## 数据与训练路线

完整合并数据集为 **108 episodes / 63,791 frames**，其中 **101 条训练示范 / 59,403 frames**，另有 **7 条保留数据 / 4,388 frames**。101 是训练集数量，不是完整数据集总量。

| 数据组 | 总 episodes | 训练 episodes | 训练 frames | 采样比例 |
|---|---:|---:|---:|---:|
| 基础示范 base | 50 | 45 | 28,083 | 30% |
| 纠错示范 recovery | 38 | 38 | 17,760 | 30% |
| 策略运行后补充 dagger | 20 | 18 | 13,560 | 40% |
| 合计 | 108 | 101 | 59,403 | 100% |

数量与 episode 映射见 [reference_split.json](dagger_training_delivery/reference_split.json)。训练数据统一为 30 Hz；数据处理与重采样入口见下方文件导航。DAgger 是本项目的数据阶段命名，不代表在此实现了自动在线专家标注系统。

训练沿同一权重链分阶段推进：基础阶段 2000 步 → 在基础权重上继续 recovery 阶段 2000 步 → 在 recovery 002000 权重上继续 DAgger 阶段 4000 步。后续阶段调整数据和超参数；本轮加载模型权重重新建立优化器，并非恢复完整训练状态。

**训练结束权重与部署选择分开记录：** DAgger 训练完成至 004000；当前根目录异步客户端校验并使用 003500。不能将当前真机结果直接标为 004000 checkpoint 的独立成绩。

## DAgger 阶段训练设置

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

采样每个 update 有整数取整，五个 update 合计精确满足 3:3:4。保留集 episode 为 0、17、26、34、44、88、106；本包未提供其独立评测结果。配置名 `pi05_wsd` 是兼容标识，实际实现为余弦调度。日志的 `cumulative_update=8000` 来自脚本固定加 4000；上面的训练路线结合实际训练记录说明，不能仅凭此字段还原权重链。

## 文件导航

- `dagger_training_delivery/`：本轮训练器、启动器、原生训练源码参考、划分及 CPU 测试。
- `pi05_full_training/`：先前基础和 recovery 训练版本，保留为训练溯源。
- `data_preparation/`、`code/rebuild_dagger_30hz.py`：数据检查和重采样。
- 根目录 `pi05_async_diag_client.py`、`pi05_rtc_server.py`、`run_rtc_server.sh`：当前异步诊断客户端与服务端入口。
- `logs_client/`：真机客户端 JSONL 运行日志。
- `rtc_compiled_delivery/`：支持 eager / compiled 的服务端，默认 eager。
- `rtc_package/`：客户端、动作队列、mock 测试及早期服务端。
- 根目录 `*benchmark*.py`、`logs/*benchmark*.json`：性能实验代码与原始结果，多个历史版本未强行合并。
- `configs/final_checkpoint/`：最终 checkpoint 的四份 JSON；推理还需 HF 上的权重和 processor safetensors。
- `results/training_metrics.csv`、`results/training_summary.json`：从日志提取的 402 条记录（非逐步完整记录）。

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

## RTC 推理与本地执行

本地采集两路图像和关节状态，异步提交云端推理请求；等待期间继续执行动作队列。服务端利用剩余动作前缀进行 RTC 引导，客户端按请求期间实际消耗的动作步数裁剪返回动作，避免重复执行旧时间片。

遇到超时、旧时间线响应或队列耗尽时，客户端保持当前位置、丢弃旧计划，再从新观测重建动作块。这里的 **通信/队列恢复** 与模型在抓取失败后发生的 **任务自主纠错** 是两件事，不能将 recovery 次数当作抓取重试次数。

在原服务器上启动与当前客户端匹配的 eager 服务端：

```bash
bash run_rtc_server.sh \
  --backend eager \
  --checkpoint /root/autodl-tmp/pi05/outputs/pi05_dagger_334_v2/checkpoints/003500/pretrained_model
```

本地通过 SSH 将服务器 8088 端口转发到 18088 后，先验证只读链路：

```bash
python pi05_async_diag_client.py --dry-run --seconds 60
```

核对串口、校准 ID、两路相机设备与起始姿态后，启用真机动作：

```bash
python pi05_async_diag_client.py --live --seconds 60
```

客户端默认 30 Hz、`request-after=1`、`initial-delay=12`，以实际日志中的参数为准。服务端启动脚本仍包含原 AutoDL 环境路径，迁移前需调整；更换 checkpoint 时需同步检查客户端的权重路径校验。旧 `rtc_package/` 与 `rtc_compiled_delivery/` 保留作历史参考。服务端无鉴权，远程访问使用 SSH 隧道，不直接暴露公网端口。

## 真机结果与延迟口径

| 指标 | 当前记录 |
|---|---|
| 测试任务 | 黄色方块抓取并放入托盘 |
| 测试次数 | 50 |
| 成功 / 失败 | 47 / 3 |
| 成功率 | **94%** |
| 测试变化 | 改变方块位置 |
| 定性观察 | 出现自主纠错行为，尚未单独统计纠错成功率 |

上述成功率来自作者真机计数；不能由训练 loss 或客户端 `merge` 数量推导。逐次成功/失败标签、超时与人工干预判定、checkpoint 和运行参数的评测归档尚待补齐。7 条离线保留数据也不等同于这 50 次真机测试。

延迟应从 [客户端日志目录](logs_client/) 单独统计，并区分：

- `request_ms`：异步请求提交到主循环处理响应的耗时，包含编码、传输、服务端处理与客户端调度等待；不包含请求提交前的观测采集。
- `response.inference_ms`：服务端预测段耗时，包括预处理、动作预测、后处理和同步，不是纯网络延迟。
- `actual_delay_steps`：请求期间已消耗的动作步数，用于动作时间线对齐。

统计时应标明具体日志、样本数、均值/P50/P95，以及是否只统计成功合并、是否包含 bootstrap 和 recovery。本 README 暂不填入尚未完整复核的端到端延迟数字，也不把下面的离线编译耗时当作真机请求延迟。

## 编译实验：速度与一致性分别报告

`logs/rtc_compile_benchmark_20261023_135706.json` 在旧 recovery checkpoint 上记录：eager 中位数 322.98 ms，compiled 146.42 ms，约 2.21×。但同时 `parity_passed=false`、`integration_candidate=false`。不能据此声称通过一致性验证或已在最终 4000 步模型上取得同等加速。服务端保留显式实验开关，默认使用 eager。

## 已运行检查

```bash
python dagger_training_delivery/test_training_logic.py
python rtc_package/test_rtc.py
```

整理时 5 项训练逻辑测试、8 项 RTC/mock 测试通过；全部 Python 通过语法解析，Shell 通过 `bash -n`。这些是此前代码整理时的静态/CPU 检查，不覆盖 GPU、摄像头或机械臂。后续作者已完成上述真机测试；本次 README 更新仅核对文档、配置和接口，未重新执行硬件测试。

## 后续补充

1. 50 次测试的逐次结果、失败案例与演示视频，以及和实际 checkpoint/运行参数的对应关系。
2. 客户端日志的可复核延迟汇总，分别报告正常合并与恢复事件。
3. 已验证的训练/推理环境导出、LeRobot Git commit 与本地改动。
4. 数据集访问方式、最初模型来源，以及 HF 权重版本和本地 checkpoint 的映射。
5. 上游代码、模型和数据的许可证及归属说明。

`results/training_summary.json` 中的 `real_robot_evaluation: pending` 是训练结束时的历史快照，不代表后续没有进行真机测试；其中 004000 的训练指标也不应与 003500 的部署记录混用。

上游项目：[Hugging Face LeRobot](https://github.com/huggingface/lerobot)、[Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)。本项目是基于上游模型与 RTC 实现的数据、训练和真机部署实践。
