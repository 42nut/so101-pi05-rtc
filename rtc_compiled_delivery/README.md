# RTC compiled 服务端

默认 Eager。Compiled 数值一致性仍未通过原阈值；参数 --allow-unverified-compile 明确开启实验模式，不代表验证通过。未修改权重、RTC guidance 或客户端动作队列。仅编译 denoise_step，模式 default。

## 安装与启动（AutoDL）

将压缩包上传到 /root/autodl-tmp/pi05，然后执行：

```bash
cd /root/autodl-tmp/pi05
unzip -n pi05_rtc_compiled_server.zip
mkdir -p logs
ss -ltnp 'sport = :8088'
pgrep -af '[p]i05_rtc_server.py'
```

先退出本地机器人控制程序。根据以上输出确认旧 RTC 服务的 PID，执行 `kill PID`（把 PID 换成实际数字），再确认 8088 已释放。不要同时运行两个服务占用 GPU。

```bash
nohup bash /root/autodl-tmp/pi05/rtc_compiled_delivery/run_rtc_server.sh \
  --backend compiled --allow-unverified-compile \
  > /root/autodl-tmp/pi05/logs/pi05_rtc_compiled_server.log 2>&1 < /dev/null &
echo "服务器 PID：$!"
tail -n 60 -f /root/autodl-tmp/pi05/logs/pi05_rtc_compiled_server.log
```

等待 SERVER_READY 后，Ctrl+C 退出 tail，不会停止 nohup 服务。首次编译预热可能数分钟。

```bash
curl --max-time 10 -sS http://127.0.0.1:8088/health
```

检查 inference_backend="compiled"、compiled=true、parity_verified=false。若为 eager，查看 compile_status.fallback_reason。ready 表示可处理请求，不表示真机验证通过。

## 本地

沿用 18088 -> 8088 SSH 隧道与原客户端。若尚未建立隧道：

```bash
ssh -p 48207 -N -L 18088:127.0.0.1:8088 root@connect.bjb1.seetacloud.com
```

另开本地终端（lerobot 环境）：

```bash
cd /home/k/soarm101/lerobot
curl --max-time 10 -sS http://127.0.0.1:18088/health
python pi05_rtc_client.py --port /dev/ttyACM1 --dry-run --seconds 20
```

dry-run 沿用原脚本的不写动作机制。查看实际 request/model 耗时、actual_delay、predicted、queue；不要直接照搬先前 146ms 基准作为服务端实测结果。预热后某些新输入路径仍可能触发重新编译；客户端超时/队列耗尽则应停止并检查日志。

无超时、无队列耗尽、确认模式仍为 compiled 后，才能比较同一起始姿态的真机表现：

```bash
python pi05_rtc_client.py --port /dev/ttyACM1 --seconds 60
```

沿用当前客户端动作限幅设置，不在本次调整。比较抓取、放置、回位、衔接停顿与任务完成时间。dry-run 通过不能替代真机验证。

## 回退

启动编译或预热失败自动恢复 Eager 并重做预热。运行中预测异常会恢复 Eager，但当前请求返回 HTTP 500，不重新生成动作；原客户端应按现有错误处理停止。先看客户端停止情况和日志，再决定重启。

手动回退：退出客户端，停止该服务 PID，使用同一启动命令把 `--backend compiled --allow-unverified-compile` 改成 `--backend eager`，然后重新启动客户端。每次服务启动生成新的 model_token，防止旧动作队列跨服务重启继续使用。

回退只处理异常，不检测静默数值误差。健康接口始终保留 parity_verified=false，不能将稳定运行当作离线数值一致性通过。

## 本次检查

已完成 Python 语法检查与模拟编译开关/回退测试。没有在交付环境运行 CUDA、真实 LeRobot 或机器人；需由上述服务端预热和本地测试验证。
