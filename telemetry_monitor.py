import argparse
from collections import deque
from contextlib import ExitStack
import csv
from dataclasses import asdict, fields
from functools import partial
import math
from pathlib import Path
from queue import SimpleQueue
import time

from cyclonedds.qos import Policy, Qos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from communicate.channel import ChannelFactoryInitialize, ChannelSubscriber
from joystick import joystick_data
from RNN_irobot import Ginger_RNN_inverse
from RNN_publisher import DDS_JOINT_SIGNS, RNN_TIMING_TOPIC, RnnTiming, th_Data


NS_PER_SECOND = 1_000_000_000
NS_PER_MS = 1_000_000
RATE_WINDOW_S = 1.0
ARMS = ("right", "left")
XYZ = "xyz"
COMMAND_FIELDS = tuple(f"target_{arm}_{axis}_mm" for arm in ARMS for axis in XYZ) + (
    "gripper_preset_1", "gripper_preset_2",
)
JOINT_FIELDS = tuple(f"{arm}_joint{index}_rad" for arm in ARMS for index in range(1, 8))
ENDPOINT_FIELDS = tuple(f"estimated_{arm}_{axis}_mm" for arm in ARMS for axis in XYZ)
STREAM_FIELDS = {"command": COMMAND_FIELDS, "joint": JOINT_FIELDS + ENDPOINT_FIELDS}
TIME_FIELDS = ("stream", "sent_unix_ns", "received_unix_ns", "sent_s", "received_s")
RNN_TIMING_FIELDS = tuple(field.name for field in fields(RnnTiming))


def window_rate(timestamps, now):
    in_window = (timestamps > now - RATE_WINDOW_S) & (timestamps <= now)
    return np.count_nonzero(in_window) / RATE_WINDOW_S


class TelemetryMonitor:
    def __init__(self, window, csv_log):
        self.window = window
        self.csv_log = csv_log
        self.pending = SimpleQueue()
        self.samples = {stream: deque() for stream in STREAM_FIELDS}
        self.timing_samples = deque()
        self.rate_history = deque()
        self.kinematics = Ginger_RNN_inverse()
        self.start_wall_ns = time.time_ns()
        self.start_monotonic_ns = time.monotonic_ns()

        self.figure = plt.figure(figsize=(12, 10), layout="constrained")
        grid = self.figure.add_gridspec(6, 2)
        self.timing_plot = self.figure.add_subplot(grid[-1, :])
        self.axes = np.array([
            [self.figure.add_subplot(grid[row, arm], sharex=self.timing_plot) for arm in range(len(ARMS))]
            for row in range(5)
        ])
        self.figure.suptitle("iRobot Telemetry")
        self.timing_plot.set(xlabel="Time since monitor start (s)", ylabel="Elapsed (ms)")
        self.iteration_line, = self.timing_plot.plot([], [], marker=".", label="Target to loop completion")
        self.last_write_line, = self.timing_plot.plot([], [], marker=".", label="Target to last joint write")
        self.timing_plot.legend(loc="upper right", fontsize="small", ncols=2)
        self.timing_plot.grid(alpha=0.25)
        self.target_lines = []
        self.endpoint_lines = []
        self.joint_lines = []
        self.rate_lines = []
        for index, arm in enumerate(ARMS):
            joint_plot, rate_plot = self.axes[-2:, index]
            self.axes[0, index].set_title(f"{arm.title()} end effector")
            joint_plot.set(title=f"{arm.title()} joint angles received", ylabel="Angle (rad)")
            rate_plot.set_ylabel("Messages/s (Hz)")
            target_lines = []
            endpoint_lines = []
            for axis_index, axis in enumerate(XYZ.upper()):
                position_plot = self.axes[axis_index, index]
                position_plot.set_ylabel(f"{axis} (mm)")
                target, = position_plot.plot(
                    [], [], ls="--", marker=".", ms=3, drawstyle="steps-post", label="Command", zorder=3,
                )
                endpoint, = position_plot.plot([], [], marker=".", ms=2, label="Estimated")
                target_lines.append(target)
                endpoint_lines.append(endpoint)
            self.target_lines.append(target_lines)
            self.endpoint_lines.append(endpoint_lines)
            self.joint_lines.append([
                joint_plot.plot([], [], marker=".", ms=2, label=f"J{i}")[0] for i in range(1, 8)
            ])
            sent, = rate_plot.plot([], [], label="DDS sent", ls="--", zorder=3)
            received, = rate_plot.plot([], [], label="Monitor received")
            self.rate_lines.append((sent, received))
            for axis, columns in ((self.axes[0, index], 2), (joint_plot, 4), (rate_plot, 2)):
                axis.legend(loc="upper right", fontsize="small", ncols=columns)
        for axis in self.axes.flat:
            axis.grid(alpha=0.25)
            axis.tick_params(labelbottom=False)

    def capture(self, stream, message):
        received_mono_ns = time.monotonic_ns()
        received_unix_ns = time.time_ns()
        self.pending.put((
            stream, message.sample_info.source_timestamp, received_unix_ns,
            received_mono_ns, message if stream == RNN_TIMING_TOPIC else tuple(message.matrix_data),
        ))

    def estimate_endpoints(self, joint_angles):
        if not np.isfinite(joint_angles).all():
            return (np.nan,) * len(ENDPOINT_FIELDS)
        right, left = (joint_angles * DDS_JOINT_SIGNS).reshape(2, 7, 1)
        right_transform = self.kinematics.Jacobian_R(right)[0]
        left_transform = self.kinematics.Jacobian_L(left)[0]
        return tuple(np.concatenate((right_transform[:3, 3], left_transform[:3, 3])) * 1000)

    def collect_samples(self):
        while not self.pending.empty():
            stream, sent_ns, received_ns, received_mono_ns, values = self.pending.get_nowait()
            sent_s = (sent_ns - self.start_wall_ns) / NS_PER_SECOND
            received_s = (received_mono_ns - self.start_monotonic_ns) / NS_PER_SECOND
            if stream == RNN_TIMING_TOPIC:
                self.timing_samples.append((received_s, values))
            else:
                if stream == "joint":
                    values += self.estimate_endpoints(values)
                self.samples[stream].append((sent_s, received_s, *values))
            if self.csv_log is not None:
                row = dict(zip(TIME_FIELDS, (stream, sent_ns, received_ns, sent_s, received_s)))
                if stream == RNN_TIMING_TOPIC:
                    row.update(asdict(values))
                    if values.last_joint_write_iteration == 0:
                        row["last_joint_write_duration_ns"] = None
                else:
                    row.update(zip(STREAM_FIELDS[stream], values))
                self.csv_log.writerow(row)

    def refresh_rnn_timing(self, plot_start):
        while self.timing_samples and self.timing_samples[0][0] < plot_start:
            self.timing_samples.popleft()
        timings = [timing for _, timing in self.timing_samples]
        target_times = [(timing.target_received_unix_ns - self.start_wall_ns) / NS_PER_SECOND for timing in timings]
        iteration_ms = [timing.iteration_duration_ns / NS_PER_MS for timing in timings]
        last_write_ms = [
            timing.last_joint_write_duration_ns / NS_PER_MS if timing.last_joint_write_iteration else np.nan
            for timing in timings
        ]
        self.iteration_line.set_data(target_times, iteration_ms)
        self.last_write_line.set_data(target_times, last_write_ms)
        title = "RNN per target | waiting for timing samples"
        if timings:
            latest = timings[-1]
            write = "no joint write"
            if latest.last_joint_write_iteration:
                write = f"last write {last_write_ms[-1]:.1f} ms at step {latest.last_joint_write_iteration}"
            title = f"RNN per target | {latest.iterations_completed} steps in {iteration_ms[-1]:.1f} ms | {write}"
        self.timing_plot.set_title(title)
        self.timing_plot.relim()
        self.timing_plot.autoscale_view(scalex=False)

    def refresh(self, _frame=None):
        self.collect_samples()
        now = (time.monotonic_ns() - self.start_monotonic_ns) / NS_PER_SECOND
        plot_start = max(0, now - self.window)
        keep_after = now - max(self.window, RATE_WINDOW_S)
        rows = {}
        for stream, history in self.samples.items():
            while history and history[0][1] < keep_after:
                history.popleft()
            rows[stream] = np.asarray(history, dtype=float).reshape(-1, 2 + len(STREAM_FIELDS[stream]))

        command, joint = rows["command"], rows["joint"]
        rate_pairs = [[window_rate(values[:, column], now) for column in (0, 1)] for values in (command, joint)]
        self.rate_history.append((now, *rate_pairs[0], *rate_pairs[1]))
        while self.rate_history[0][0] < plot_start:
            self.rate_history.popleft()
        rates = np.asarray(self.rate_history)

        target_xyz = command[:, 2:2 + len(ENDPOINT_FIELDS)].reshape(-1, 2, 3)
        joint_xyz = joint[:, -len(ENDPOINT_FIELDS):].reshape(-1, 2, 3)
        angles = joint[:, 2:2 + len(JOINT_FIELDS)].reshape(-1, 2, 7)
        for arm in range(len(ARMS)):
            for axis in range(len(XYZ)):
                self.target_lines[arm][axis].set_data(command[:, 0], target_xyz[:, arm, axis])
                self.endpoint_lines[arm][axis].set_data(joint[:, 1], joint_xyz[:, arm, axis])
            for index, line in enumerate(self.joint_lines[arm]):
                line.set_data(joint[:, 1], angles[:, arm, index])
            for index, line in enumerate(self.rate_lines[arm]):
                line.set_data(rates[:, 0], rates[:, 1 + 2 * arm + index])
            sent, received = rate_pairs[arm]
            label = ("End-effector commands", "Joint angles")[arm]
            self.axes[-1, arm].set_title(f"{label} | sent {sent:.0f} Hz, received {received:.0f} Hz")

        self.refresh_rnn_timing(plot_start)
        self.timing_plot.set_xlim(plot_start, max(now, self.window))
        for axis in self.axes.flat:
            axis.relim()
            axis.autoscale_view(scalex=False)
        for axis in self.axes[-1]:
            axis.set_ylim(0, max(1, axis.get_ylim()[1]), auto=True)


def main():
    parser = argparse.ArgumentParser(description="Monitor iRobot DDS commands, joint angles and rates")
    parser.add_argument("--window", type=float, default=15, help="time shown on the plots in seconds (default: 15)")
    parser.add_argument("--csv", type=Path, help="record all received samples to a new CSV file")
    args = parser.parse_args()
    if not math.isfinite(args.window) or args.window <= 0:
        parser.error("--window must be a positive, finite number")

    ChannelFactoryInitialize()
    with ExitStack() as stack:
        csv_log = None
        if args.csv is not None:
            try:
                output = stack.enter_context(args.csv.open("x", encoding="utf-8", newline="", buffering=1))
            except OSError as error:
                parser.error(str(error))
            csv_log = csv.DictWriter(
                output, fieldnames=TIME_FIELDS + COMMAND_FIELDS + JOINT_FIELDS + ENDPOINT_FIELDS + RNN_TIMING_FIELDS,
            )
            csv_log.writeheader()
            print(f"Recording DDS samples to {args.csv}")

        monitor = TelemetryMonitor(args.window, csv_log)
        stack.callback(monitor.collect_samples)
        qos = Qos(Policy.History.KeepLast(512))
        timing_qos = Qos(Policy.History.KeepLast(512), Policy.Reliability.BestEffort)
        streams = (
            ("command", "topic", joystick_data, qos),
            ("joint", "topic", th_Data, qos),
            (RNN_TIMING_TOPIC, RNN_TIMING_TOPIC, RnnTiming, timing_qos),
        )
        for stream, topic, message_type, reader_qos in streams:
            subscriber = ChannelSubscriber(topic, message_type)
            subscriber.Init(partial(monitor.capture, stream), qos=reader_qos)
            stack.callback(subscriber.Close)

        animation = FuncAnimation(monitor.figure, monitor.refresh, interval=200, cache_frame_data=False)
        try:
            plt.show()
        except KeyboardInterrupt:
            plt.close(monitor.figure)
        finally:
            if animation.event_source is not None:
                animation.event_source.stop()


if __name__ == "__main__":
    main()
