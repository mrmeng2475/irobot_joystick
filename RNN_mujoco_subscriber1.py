import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


SCENE_PATH = Path(__file__).resolve().parent / "asset" / "scene_irobot.xml"
ARM_JOINTS = [f"joint{index}_{side}" for side in ("R", "L") for index in range(1, 8)]


def validate_positions(positions, joint_range):
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (len(ARM_JOINTS),) or not np.isfinite(positions).all():
        raise ValueError("Expected 14 finite joint angles")
    if np.any(positions < joint_range[:, 0]) or np.any(positions > joint_range[:, 1]):
        raise ValueError("Target positions out of joint range")
    return positions


def main():
    parser = argparse.ArgumentParser(description="Run the iRobot MuJoCo simulation")
    parser.add_argument("--dds", action="store_true", help="Follow the joint angles published by RNN_publisher.py")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
    actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINTS]
    joint_range = model.jnt_range[joint_ids]
    target_positions = np.zeros(len(ARM_JOINTS))
    subscriber = None

    try:
        if args.dds:
            from communicate.channel import ChannelFactoryInitialize, ChannelSubscriber
            from RNN_publisher import th_Data

            ChannelFactoryInitialize()
            subscriber = ChannelSubscriber("topic", th_Data)
            subscriber.Init()

        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0, 0, 0.8]
                viewer.cam.distance = 2.0
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -20.0

            while viewer.is_running():
                if subscriber is not None:
                    message = subscriber.Read(0)
                    if message is not None:
                        try:
                            target_positions = validate_positions(message.matrix_data, joint_range)
                        except ValueError as error:
                            print(f"Invalid DDS joint target: {error}")

                data.ctrl[actuator_ids] = target_positions
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        print("Simulation stopped")
    finally:
        if subscriber is not None:
            subscriber.Close()


if __name__ == "__main__":
    main()
