# π0.5 全量训练启动包

适配你提供的 LeRobot 0.6.1 训练源码、单张 A800，以及当前 SO101 黄方块入托盘数据。脚本校验训练函数的 SHA256，不修改 site-packages。请保留当前能跑通 benchmark 的环境。

## 固定配置

| 项目 | 设置 |
|---|---|
| 微批次 / 梯度累积 / 有效批次 | 8 / 32 / 256 |
| 精度 / 梯度检查点 | BF16 / 关闭 |
| 训练范围 | 全量参数，不冻结视觉编码器，不使用 LoRA |
| 优化器 | AdamW，峰值 LR 5e-5，betas=(0.9,0.95)，weight decay=0.01，梯度裁剪=1 |
| 数据 | 原定 45 个训练 episode；保留 [0,17,26,34,44] 不参与训练 |
| 动作 | 绝对动作；状态和动作 QUANTILES，视觉 IDENTITY |
| 保存 | 每 500 次优化器更新；暂停时额外保存；默认保留最近 2 份完整检查点 |

学习率采用参考 Dream Machines 思路制定的 WSD 配置，并非照搬其全部实验参数：前 100 次更新线性升到 5e-5，保持到第 24,000 次，再以二次曲线下降，到第 30,000 次为 5e-6。`--until` 只决定本次暂停点，不改变这条曲线。30,000 是调度长度，不要求现在跑满。

本包不自动计算保留集 loss，也不执行机器人 rollout。先到 1,000 步检查效果，再决定是否继续。训练 loss 下降本身不能证明真实抓取成功率提升。

## 启动

把压缩包上传到 `/root/autodl-tmp/pi05`，执行：

```bash
cd /root/autodl-tmp/pi05
unzip pi05_full_training.zip
bash pi05_full_training/run_pi05_full.sh --check-only
```

看到 `PREFLIGHT PASSED` 后，先跑两步真实训练并保存：

```bash
bash pi05_full_training/run_pi05_full.sh --until 2
```

看到 `FINISHED AND SAVED at optimizer step 2` 后，从这份检查点续训到累计 1,000 步：

```bash
mkdir -p /root/autodl-tmp/pi05/logs
nohup bash /root/autodl-tmp/pi05/pi05_full_training/run_pi05_full.sh \
  --resume --until 1000 \
  > /root/autodl-tmp/pi05/logs/pi05_full_wsd_v1.log 2>&1 &
tail -f /root/autodl-tmp/pi05/logs/pi05_full_wsd_v1.log
```

`tail -f` 可以 Ctrl+C 退出，不影响 nohup 训练进程。不要同时启动两个训练进程。

按已测的约 21.03 秒/更新，累计 1,000 步约需 5.84 小时，另加加载和保存时间。两步测试只有约 42 秒的更新计算，但模型加载与完整检查点保存另计。

## 检查点和恢复

默认输出：`/root/autodl-tmp/pi05/outputs/pi05_full_wsd_v1`。

- 推理模型和处理器：`checkpoints/last/pretrained_model/`。
- 续训状态：检查点内原生 `training_state/`，包括原生保存的优化器、调度器和 RNG 状态。
- 更新日志：输出目录中的 `updates.jsonl`，包含实际使用的学习率、loss、梯度范数、耗时和峰值分配显存。
- 数据恢复位置按每次更新 32 个微批次计算，并计入每轮丢弃的不完整批次。GPU 运算等因素仍可能影响逐位复现。

必须通过本包的启动脚本恢复，不能直接使用普通 `lerobot-train`：本包注册了专用 WSD，并补充了梯度累积和数据位置恢复逻辑。恢复会核对配置、脚本、数据元信息和统计文件；不要在同一轮训练中随意修改它们。

例如已经保存到 1,000 步，后续决定继续到累计 3,000 步时：

```bash
bash /root/autodl-tmp/pi05/pi05_full_training/run_pi05_full.sh --resume --until 3000
```

每次先写入临时目录，成功后再更新 last，随后清理较旧的本包检查点。保留 2 份时，保存过程中磁盘可能同时存在 3 份。全量检查点含优化器，可能达到数十 GB；脚本会按实际张量大小检查保存空间，恢复时也会估算后续保存空间。

若首次运行失败且没有完整检查点，不能使用 `--resume`。可指定新的 `--output-dir` 重试，之后恢复时继续传入同一个路径。脚本不会自动删除失败运行的输出目录。

## 验证范围

本地 6 项检查已通过：原生源码匹配和类型注解、累积梯度平均与单次更新、原生循环暂停/续训保存、丢弃尾批后的采样恢复、WSD 边界、原子保存失败处理和检查点保留。

这些是控制流程及数学逻辑检查，本地没有你的 CUDA/LeRobot 运行环境。上面的两步保存及随后的续训用于验证服务器上的真实模型、原生保存函数和恢复流程；尚未声称完成 A800 实机验证。
