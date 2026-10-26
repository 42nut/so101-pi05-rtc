import time

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

# 之前的串口配置
LEADER_PORT = "/dev/ttyACM0"
FOLLOWER_PORT = "/dev/ttyACM1"

MOTOR_ID = 3

def create_bus(port):
    return FeetechMotorsBus(
        port=port,
        motors={
            "elbow_flex": Motor(
                MOTOR_ID,
                "sts3215",
                MotorNormMode.DEGREES,
            )
        },
    )

leader = create_bus(LEADER_PORT)
follower = create_bus(FOLLOWER_PORT)

try:
    leader.connect()
    follower.connect()

    # 只读位置，不发送运动指令
    print("开始读取肘关节...")
    print("请缓慢移动主臂和从臂，观察编码器读数。")

    start_l = leader.read(
        "Present_Position",
        "elbow_flex",
        normalize=False
    )

    start_f = follower.read(
        "Present_Position",
        "elbow_flex",
        normalize=False
    )

    while True:

        pos_l = leader.read(
            "Present_Position",
            "elbow_flex",
            normalize=False
        )

        pos_f = follower.read(
            "Present_Position",
            "elbow_flex",
            normalize=False
        )

        delta_l = (pos_l - start_l) * 360 / 4096
        delta_f = (pos_f - start_f) * 360 / 4096

        print(
            f"主臂: {delta_l:+7.2f}° | "
            f"从臂: {delta_f:+7.2f}° | "
            f"差值: {delta_l-delta_f:+7.2f}°"
        )

        time.sleep(0.3)

except KeyboardInterrupt:
    print("\n测试结束")

finally:
    if leader.is_connected:
        leader.disconnect()
    if follower.is_connected:
        follower.disconnect()
