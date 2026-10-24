# π0.5 RTC 接入（LeRobot 0.6.1）

针对本次上传的 RTC 源码和本地控制脚本编写。使用已训练的 pi05_recovery_cosine_tc_v1/002000 权重，无需重新训练。

## 实现范围

- 服务器调用安装版本的 RTCProcessor 引导采样，传入 prev_chunk_left_over、inference_delay、execution_horizon，是真正的 RTC 引导，不只是异步预取。
- 服务器同时返回反归一化动作和归一化动作；本地原样保留归一化尾部，下次传回。6 维前缀由官方 RTCProcessor 补零至内部 32 维。
- use_relative_actions 必须为 False，沿用 checkpoint 预处理器和反归一化器。
- RTC 内部需要 enable_grad，外层采用 no_grad，禁止 inference_mode。
- 第一版采用 eager（compile_model=False），先验证 RTC 正确性；不能沿用先前编译式同步服务的 76ms 速度预期。
- 30Hz 动作执行，50 步预测；默认当前块 cursor=20 时发起下次请求，还剩 30 步约 1 秒覆盖推理。它不再每次执行固定 30 步：若请求约耗 10 步，则通常约在原块第 30 步附近衔接。
- 请求预计延迟：至少 12 步，或最近 20 次完整请求最大延迟向上取整再加 2 步。预计延迟必须小于剩余队列。
- 新预测以请求时观测为时间起点。收到响应时按实际已消费动作数量裁掉前缀。快回复可以提前衔接，慢回复不能重复旧动作。
- 只有主线程操作串口/相机/动作队列；后台线程仅做图片编码和 HTTP。一时只允许一个请求在途。
- 错误身份、响应超期、队列耗尽、HTTP 异常均停止发送动作并尝试保持当前姿态，不自动重新启动。
- 相机读取本身仍在主线程，串口和相机耗时可能造成 tick 抖动。记录 tick_late_ms；本实现不是硬实时保证。
- 本地保留 slew=10、max_relative_target=20.0 和关节归一化边界。RTC 引导的是模型计划前缀，实际动作被大幅限幅时不能保证计划与实机一致；日志保存 requested/sent 便于判断。

## 安装

本地：将下载的 pi05_rtc_bundle.zip 放到 ~/Downloads，执行：

```bash
conda activate lerobot
unzip ~/Downloads/pi05_rtc_bundle.zip -d ~/soarm101/lerobot
```

上传整个包到服务器，端口若变化需替换：

```bash
scp -P 48207 ~/Downloads/pi05_rtc_bundle.zip root@connect.bjb1.seetacloud.com:/root/autodl-tmp/pi05/
```

服务器：

```bash
cd /root/autodl-tmp/pi05
unzip pi05_rtc_bundle.zip
mkdir -p logs
nohup bash rtc_package/run_rtc_server.sh > logs/pi05_rtc_server.log 2>&1 < /dev/null &
echo "RTC server PID: $!"
tail -n 60 -f logs/pi05_rtc_server.log
```

仅启动一次。新端口 8088，旧同步服务 8087 不受影响。若 GPU 显存不足，先停止旧同步服务再启动 RTC，不要同时重复加载多个 RTC 服务。

等待 WARMUP guided=1/2/3 和 SERVER_READY。预热包含真实 RTC 前缀调用，若该路径失败，服务不会就绪。

```bash
curl --max-time 10 -sS http://127.0.0.1:8088/health
```

本地另开终端建立转发，保持打开：

```bash
ssh -p 48207 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -N -L 18088:127.0.0.1:8088 root@connect.bjb1.seetacloud.com
```

## 第一次：只读机器人，不发送动作

关闭其他本地遥操作/控制程序；校准使用之前的 my_awesome_follower_arm。串口仍需是实际从臂端口。

```bash
conda activate lerobot
cd ~/soarm101/lerobot/rtc_package
python pi05_rtc_client.py --port /dev/ttyACM1 --dry-run --seconds 20
```

此模式仍连接设备、读取相机，机器人连接本身可能使电机使能；脚本不会调用 send_action，包括退出阶段。它模拟队列消费，只验证网络、RTC 前缀和队列，不证明真机跟踪或抓取效果。

应连续出现 RTC merge，且无 error。检查请求延迟应明显低于剩余队列约 1 秒预算。如果预计延迟超过预算或发生超期，程序会停止；先看日志，不要直接延长 timeout 或解除保护。

## 真机

只读联调通过后，将方块与托盘摆在之前成功的位置。按 Ctrl+C 停止；HTTP 在后台，不影响主线程处理 Ctrl+C，串口调用本身仍可能阻塞。

```bash
python pi05_rtc_client.py --port /dev/ttyACM1 --seconds 60 --slew 10 --max-relative-target 20.0
```

完成放置回位即停止，程序不自动识别任务成功。退出保持力矩，不要直接强扳机械臂。日志在 rtc_package/logs/pi05_rtc_*.jsonl。

## 验证范围

在开发环境通过 8 项测试：实际延迟 0/3/9/17/29 步对齐、多次合并无时间跳回、重复/旧代响应拒绝、队列耗尽、NaN 拒绝且保持队列不变、dry-run 零动作写入、模拟实机单线程硬件所有权、错误响应停止并断开。可复跑：

```bash
python test_rtc.py
```

未在用户 A800 / SO101 上执行真实 RTC 推理或物理测试。必须以服务器 guided 预热、20 秒 dry-run 和真机日志为准。文件中的内部条件检查会在接口不匹配时停止。

## 文件

- pi05_rtc_server.py：GPU 模型及 RTC 引导 HTTP 服务。
- pi05_rtc_client.py：本地异步执行、观测、限幅与日志。
- rtc_queue.py：按实际消费步数对齐的动作队列。
- run_rtc_server.sh：独立环境及动态库启动设置。
- test_rtc.py：无需 GPU / 机械臂的队列及模拟联调测试。
