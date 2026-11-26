# iRobot 双臂手柄遥操作

本项目通过手柄设置双臂末端的目标位置，用 RNN 迭代求解逆运动学，并通过 Cyclone DDS 发布每臂 7 个关节的目标角度。项目提供 iRobot 双臂的 MuJoCo 模型，以及通过飞特串口舵机驱动右臂和夹爪的脚本。

## 快速开始

以下命令以 Linux 为例，请在项目根目录执行。MuJoCo 查看器需要能够打开图形窗口的桌面环境。

如果已创建 Python 3.10 的 `irobot_joystick` Conda 环境，可以一次装好各脚本的依赖。

```bash
conda activate irobot_joystick
python -m pip install numpy matplotlib pygame mujoco cyclonedds pyserial keyboard
```

也可以用 `venv` 创建一个环境，先安装 MuJoCo 预览所需的依赖。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy mujoco
```

激活你选择的环境后运行。

```bash
python 1.py
```

`1.py` 加载 `asset/scene_irobot.xml` 并运行物理仿真，可单独启动查看模型。关闭查看器窗口即可退出。

## 手柄与逆运动学

激活虚拟环境后，可先运行独立的逆运动学示例。它使用内置的目标位置求解，输出计算出的末端位置和耗时。

```bash
python -m pip install numpy matplotlib
python RNN_irobot.py
```

手柄采集和消息传输还需要 Pygame 和 Cyclone DDS。

```bash
python -m pip install pygame cyclonedds
```

若 `cyclonedds` 需要从源码安装，可参考 [Cyclone DDS 官方安装文档](https://cyclonedds.io/docs/cyclonedds/latest/getting_started/installation/index.html)。

连接手柄后，在两个终端进入项目根目录，并分别激活同一个 Python 环境。

1. 在终端一运行 `python RNN_publisher.py`，等待手柄消息。
2. 在终端二运行 `python joystick.py`，开始发送目标位置。

发布关节角度还需要一个正在接收 `th_Data` 的 DDS 订阅端。出现 `Waiting for subscriber.` 时，检查对应订阅端是否在运行，以及各进程是否处于同一个 DDS 域并能通过所选网卡发现彼此。脚本默认使用域 `0`，由 Cyclone DDS 自动选择网卡。

### 手柄映射

脚本使用 Pygame 识别到的第一个手柄，需要至少 6 个轴和 6 个按钮。编号从 `0` 开始，具体对应的物理按键取决于手柄型号。

| 操作 | 按代码使用的 Pygame 编号 |
| --- | --- |
| 右臂沿 X / Y 移动 | 轴 `1` / `0` |
| 右臂下降 / 上升 | 轴 `2` / 按钮 `4` |
| 左臂沿 X / Y 移动 | 轴 `4` / `3` |
| 左臂下降 / 上升 | 轴 `5` / 按钮 `5` |
| 重置双臂末端目标 | 按钮 `2` |
| 切换真机夹爪的两个预设位置 | 按钮 `0` / `1` |

### DDS 消息

两类消息使用同一个主题名 `topic`，都通过 `matrix_data` 传递数据。

| 类型 | 发布脚本 | 数据顺序 |
| --- | --- | --- |
| `joystick_data` | `joystick.py` 或 `keyboard_teleop.py` | 右臂末端 XYZ、左臂末端 XYZ（单位为毫米），再接两个夹爪预设的触发状态 |
| `th_Data` | `RNN_publisher.py` | 右臂关节 1–7、左臂关节 1–7（单位为弧度） |

## 键盘遥控

在三个终端分别激活同一个 Python 环境，然后依次运行下面的命令，每个终端运行一条。

```bash
python RNN_mujoco_subscriber1.py --dds
python RNN_publisher.py
python keyboard_teleop.py
```

点击键盘遥控窗口并按住按键。移动键每次更新将对应目标坐标增减 10 mm，窗口显示两臂当前的目标坐标。窗口失去焦点时会保留当前目标。

| 操作 | 正方向 / 负方向 |
| --- | --- |
| 右臂 X / Y / Z | `W` / `S`，`A` / `D`，`Q` / `E` |
| 左臂 X / Y / Z | `I` / `K`，`J` / `L`，`U` / `O` |
| 真机夹爪的两个预设 | `1` / `2` |
| 重置双臂末端目标 | `Space` |
| 退出键盘遥控 | `Esc` |

手柄和键盘会发布同一种消息。联调时选择一个输入脚本运行。

## 实时监控曲线

在另一个终端激活 `irobot_joystick`，和遥控脚本一起运行监控器。

```bash
python telemetry_monitor.py
```

- 最上面三排分别画双臂的 X、Y、Z。虚线上的点是发送的末端目标，实线是根据收到的关节目标角度用正运动学推算的末端点位，单位为毫米。
- 第四排画收到的右臂和左臂各 7 个关节角，单位为弧度。横轴单位为秒，以启动监控为起点。末端指令按 DDS 发送时间绘制，关节角和推算点位按监控器接收时间绘制。
- 第五排对末端指令和关节角分别显示发送与接收频率，按最近 1 秒内监控捕获的样本统计，断流后归零。发送时间来自 DDS，接收时间是监控器读到样本的时刻。
- 最下方显示每个目标从 RNN 收到到本轮结束、以及到最后一次成功写入关节角的耗时，单位为毫秒。两条曲线以 RNN 收到该目标的时刻为横坐标。RNN 每轮计划迭代 300 步，关节角按 60 Hz 节流，最后一次写入可能早于第 300 步。如果本轮没有成功写入，发布曲线留空。

计时包含循环中的日志和 DDS 写入等待。跨机器运行时需要同步系统时间来对齐图上的横坐标，每轮耗时在 RNN 进程内用单调时钟计算。

默认显示最近 15 秒。下面的命令将窗口改为 30 秒，并将整次运行捕获的样本写到新建 CSV，记录发送和接收的 Unix 纳秒、双臂末端目标和推算点位、夹爪指令及 14 个关节角。

```bash
python telemetry_monitor.py --window 30 --csv telemetry.csv
```

CSV 中 `rnn_timing` 行还记录原目标的发送和 RNN 接收时间、实际迭代步数、最后一次成功写入的步数及对应耗时。更新后需重启 `RNN_publisher.py` 和监控器以启用此曲线。

## MuJoCo 仿真

直接运行时，将 14 个关节的目标角度设为 `0`。

```bash
python RNN_mujoco_subscriber1.py
```

加上 `--dds` 后，仿真会读取 `RNN_publisher.py` 发布的关节角度。收到有效角度就更新目标，等待消息时继续执行仿真。

## 飞特舵机真机

真机脚本使用仓库内的 SCServo SDK，串口默认为 `/dev/ttyUSB0`，波特率为 `1000000`。舵机 ID `1`–`7` 接收 `th_Data` 中的右臂关节角度，ID `8` 对应夹爪。

**`FeiteCtl.py` 启动后会立即写入舵机初始位置。** 运行前请核对串口访问权限、舵机 ID、关节方向和初始姿态，并留出机械臂的运动空间。脚本直接读取 DDS 消息字段，遇到空消息可能退出。

```bash
python -m pip install pyserial keyboard
python FeiteCtl.py
```

真机联调时同时运行 `RNN_publisher.py` 和一个输入脚本（`joystick.py` 或 `keyboard_teleop.py`）。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `joystick.py` | 读取手柄轴和按钮，发布双臂末端目标 |
| `keyboard_teleop.py` | 在独立窗口读取键盘，发布双臂末端目标 |
| `telemetry_monitor.py` | 实时显示 DDS 指令、关节角、末端点位和频率，可记录 CSV |
| `RNN_irobot.py`、`RNN_publisher.py` | 迭代求解双臂逆运动学，发布关节角度 |
| `1.py`、`RNN_mujoco_subscriber1.py` | MuJoCo 模型预览与控制入口 |
| `FeiteCtl.py`、`SCServo_Python/` | 飞特舵机控制脚本、SDK 和示例 |
| `communicate/` | Cyclone DDS 通信封装和网卡配置 |
| `asset/` | iRobot MuJoCo 场景、机器人模型与 STL 网格 |
| `ginger_mujoco/` | Ginger 的 MuJoCo 场景、URDF/Xacro 和 ROS 启动文件 |

## 演示录像

- [仿人.mkv](录制/仿人.mkv)
- [无仿人.mkv](录制/无仿人.mkv)
