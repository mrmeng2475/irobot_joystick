from contextlib import ExitStack
from dataclasses import dataclass
import time

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, int32, int64
from cyclonedds.qos import Policy, Qos
import numpy as np

from communicate.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from joystick import joystick_data
from RNN_irobot import Ginger_RNN_inverse


DDS_JOINT_SIGNS = np.array([1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
ITERATIONS_PER_TARGET = 3000
SOLVER_STEP_SECONDS = 0.00012
JOINT_PUBLISH_PERIOD_NS = 1_000_000_000 / 60
RNN_TIMING_TOPIC = "rnn_timing"


@dataclass
class th_Data(IdlStruct, typename="th_Data"):
    matrix_data: array[float, 14]


@dataclass
class RnnTiming(IdlStruct, typename="RnnTiming"):
    target_sent_unix_ns: int64
    target_received_unix_ns: int64
    iterations_completed: int32
    iteration_duration_ns: int64
    last_joint_write_iteration: int32
    last_joint_write_duration_ns: int64


def main():
    ChannelFactoryInitialize()

    with ExitStack() as channels:
        sub = ChannelSubscriber("topic", joystick_data)
        sub.Init()
        channels.callback(sub.Close)
        print("等待接收手柄信息")

        pub = ChannelPublisher("topic", th_Data)
        pub.Init()
        channels.callback(pub.Close)

        timing_pub = ChannelPublisher(RNN_TIMING_TOPIC, RnnTiming)
        timing_pub.Init(qos=Qos(Policy.Reliability.BestEffort))
        channels.callback(timing_pub.Close)

        solver = Ginger_RNN_inverse(dt=SOLVER_STEP_SECONDS)
        velocity = np.zeros((3, 1))
        th_r = np.zeros((7, 1))
        th_l = np.zeros((7, 1))
        last_publish_ns = time.perf_counter_ns()

        try:
            while True:
                target = sub.Read()
                target_received_ns = time.perf_counter_ns()
                target_received_unix_ns = time.time_ns()
                if target is None or not hasattr(target, "matrix_data") or len(target.matrix_data) < 6:
                    print("No valid data received")
                    time.sleep(0.01)
                    continue

                pos_r = np.array(target.matrix_data[:3]).reshape(3, 1)
                pos_l = np.array(target.matrix_data[3:6]).reshape(3, 1)
                print(f"Received pos_R: {pos_r}, pos_L: {pos_l}")

                iterations_completed = 0
                last_write_iteration = 0
                last_write_duration_ns = -1
                try:
                    generator = solver.Inverse_with_time(
                        pos_r, pos_l, velocity, velocity, th_r, th_l,
                        iterations=ITERATIONS_PER_TARGET,
                    )
                    for iterations_completed, (angles, _) in enumerate(generator, start=1):
                        now_ns = time.perf_counter_ns()
                        if now_ns - last_publish_ns >= JOINT_PUBLISH_PERIOD_NS:
                            command = th_Data(matrix_data=angles.flatten() * DDS_JOINT_SIGNS)
                            if pub.Write(command, 0.5):
                                last_write_duration_ns = time.perf_counter_ns() - target_received_ns
                                last_write_iteration = iterations_completed
                                last_publish_ns = now_ns
                                print("Publish success. msg:", command.matrix_data)
                            else:
                                print("Waiting for subscriber.")
                        th_r = angles[:7].reshape(7, 1)
                        th_l = angles[7:14].reshape(7, 1)
                except Exception as error:
                    print(f"Error in Inverse_with_time: {error}")

                iteration_duration_ns = time.perf_counter_ns() - target_received_ns
                timing_pub.Write(RnnTiming(
                    target_sent_unix_ns=target.sample_info.source_timestamp,
                    target_received_unix_ns=target_received_unix_ns,
                    iterations_completed=iterations_completed,
                    iteration_duration_ns=iteration_duration_ns,
                    last_joint_write_iteration=last_write_iteration,
                    last_joint_write_duration_ns=last_write_duration_ns,
                ))
        except KeyboardInterrupt:
            print("程序终止")


if __name__ == "__main__":
    main()
