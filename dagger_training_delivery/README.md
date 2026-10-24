# π0.5 DAgger 3:3:4 全量微调

## 使用
将目录解压到 /root/autodl-tmp/pi05/dagger_training_delivery。
先结束占用 GPU 的推理服务，检查 nvidia-smi；启动器不会自动终止其他进程。

```bash
cd /root/autodl-tmp/pi05/dagger_training_delivery
python test_training_logic.py
bash start_training.sh --until 1000 --save-every 500 --keep-checkpoints 4
```
start_training.sh 先运行预检，通过才 nohup 启动，输出独立日志路径和 PID。
预检不加载 4B 模型，不创建训练输出；读取真实数据、解析配置、验证采样池并用 TorchCodec 解码。
日志出现 FULL_UPDATE 表示实际优化器更新已完成；仅看到后台 PID 不代表训练成功。

续训（同一目录、脚本及 split 保持不变）：
```bash
bash start_training.sh --resume --until 2000 --save-every 500 --keep-checkpoints 4
# 评估后需要继续时：
bash start_training.sh --resume --until 4000 --save-every 500 --keep-checkpoints 4
```
--until 是本轮绝对更新数，不是增加的步数。新训练 1000 对应累计 5000。
--resume 只能用于本轮输出，不能用于导入上一轮 recovery 训练状态。

## 固定配置
- 全量训练，BF16，micro=8，累积=32，有效 batch=256，单 GPU。
- 从 outputs/pi05_recovery_cosine_tc_v1/checkpoints/002000/pretrained_model 加载模型和归一化处理器。
- 新建 AdamW，lr=1e-5，betas=(0.9,0.95)，eps=1e-8，weight_decay=0.01，clip=1。
- 本轮完整调度4000步，warmup100，余弦下降到1e-6。--until1000只暂停，不提前衰减到最低。
- 默认输出 outputs/pi05_dagger_334_v1，不覆盖任何旧训练。
- TorchCodec，num_workers=0，禁用增强、编译、梯度检查点、PEFT、relative actions。
- 所有样本使用此前统一的抓取放置任务文本。
- 保留最近4个完整checkpoint；保存下一个时会临时多占一份空间。仅清理本轮输出中的完整checkpoint。

## 采样及断点
101条训练、59403帧；7条验证不参与训练：0,17,26,34,44,88,106。
基础45条28083帧；恢复38条17760帧；DAgger18条13560帧。
三组内按训练帧均匀、有放回抽取。整数计数每5次更新精确为384/384/512；单次更新是76/76/104或77/77/102，随机打散后切为32个micro batch。
保留模型原生drop_n_last_frames语义（如果配置非零），先剔除尾部再分组抽样。
采样器用过滤后数据集的局部索引，避免直接拿合并后的全局帧号索引子集。
每个update由(seed,update)独立生成样本顺序，不依赖全局随机数或预取游标。
checkpoint中保存下一次更新的样本位置；续训使用当前checkpoint处理器和原生优化器、调度器、RNG恢复逻辑。
FULL_UPDATE及updates.jsonl包含group_samples和cumulative_update，实际batch组成不符会报错。
日志里的epoch来自原生tracker，对有放回分组采样只作样本量参考，不代表无重复完整遍历。

## 验证范围
本地通过5项CPU逻辑测试：4000步比例、断点样本顺序、过滤后索引、学习率端点、原生trainer AST及划分检查；Python语法、shell语法通过。
本交付环境没有Torch/CUDA、真实数据或权重，未完成实际GPU训练；服务器预检和首个更新是必需的现场验证。
验证集被排除，但当前脚本不自动计算验证loss（沿用eval_steps=0）；保存checkpoint后需另行评估，不能把训练loss当成验证成绩。
reference_native_train.py/reference_split.json仅用于测试；训练仍读取服务器安装的trainer和实际split并校验。
