import time
import numpy as np
import mujoco
import mujoco.viewer

# 初始化 MuJoCo
model = mujoco.MjModel.from_xml_path("./asset/scene_irobot.xml")
data = mujoco.MjData(model)

# 定义右臂关节名称
arm_joints = [
    "joint1_R", "joint2_R", "joint3_R",
    "joint4_R", "joint5_R", "joint6_R", "joint7_R",
    "joint1_L", "joint2_L", "joint3_L",
    "joint4_L", "joint5_L", "joint6_L", "joint7_L"
]

# 获取关节和执行器索引
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in arm_joints]
actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in arm_joints]

# 验证执行器数量
assert len(actuator_ids) == len(arm_joints), "执行器数量与关节数量不匹配"

# 获取关节角度范围
joint_range = model.jnt_range[joint_ids]

# 初始化目标关节角度
target_positions = np.zeros(14)

def generate_random_positions():
    """生成随机的关节角度，确保在关节范围内"""
    positions = np.zeros(14)
    for i in range(14):
        low, high = joint_range[i]
        positions[i] = np.random.uniform(low, high)
    return positions

if __name__ == "__main__":
    # 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 配置初始视角
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20.0

        # 记录仿真开始时间
        time1 = time.time()
        last_update = time1
        try:
            while viewer.is_running():
                # 每隔2秒生成新的随机目标位置
                # current_time = time.time()
                # if current_time - last_update > 2.0:
                #     target_positions = generate_random_positions()
                #     print(f"New target positions: {target_positions}")
                #     last_update = current_time

                # 设置控制信号
                # data.ctrl[actuator_ids] = target_positions

                # 执行物理仿真
                mujoco.mj_step(model, data)

                # 更新查看器
                viewer.sync()

                # 手动控制频率 (~60Hz)
                time.sleep(1/60)

        except KeyboardInterrupt:
            print("Simulation stopped")