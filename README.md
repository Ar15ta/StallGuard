# StallGuard · 轴流压气机失速预警原型

面向轴流压气机旋转失速先兆的**在线因果预警**原型程序。工作方式：

1. 用一段已知正常工况数据做**初始化标定**，冻结每个探针的稳健基线（中位数 + 1.4826×MAD）。
2. 程序监听 `127.0.0.1:8765`，通过 TCP 接收实时多探针压力数据。
3. 每个分析窗计算所选机器判据（时序自相关 / 排列熵 / 小波细节能量比）相对基线的标准化偏差，得到风险分数。
4. 逐探针滞回状态机 + K-of-N 确认给出系统报警，界面点亮红灯并记录事件。
5. 附带失速团周向定位与 16 通道主动控制接口（当前为软件回调与动画，未绑定真实硬件）。

另提供**传感器模拟器**，把 CSV 按指定速率经 TCP 发进主程序，无需真实实验台即可完整跑通"标定 → 监听 → 失速识别 → 导出"演示。

> **AI 辅助生成声明**
> 本项目的源代码、界面、测试与文档由作者在 AI 编程助手（TraeCode）辅助下生成、重构与整理；作者负责需求定义、判据选型、参数标定与结果验证。请自行复核数值结果，本项目是实验与工程演示原型，**不是**经过认证的安全联锁系统。

---

## 1. 快速开始

要求 Python 3.11 及以上（开发环境为 3.14），Windows 下已验证。

```powershell
git clone https://github.com/Ar15ta/StallGuard.git
cd StallGuard\source
python -m pip install -r requirements.txt
python app.py
```

README 中所有命令都以 `source\` 为工作目录；`..\data\` 即仓库顶层的数据目录。

### 跑一次完整演示

需要两个终端。

**终端 A —— 预警主程序**

```powershell
cd source
python app.py
```

启动后：

1. 左侧「机器判据」选择判据（默认“时序自相关”）。
2. 点击「初始化标定」，选择 `..\data\pressure_u1_initialization.csv`。
3. 确认对话框中的采样间隔与探针映射（默认值即该数据文件同源参数），等待状态变为“正在监听 127.0.0.1:8765”。

**终端 B —— 传感器模拟器**

图形界面（推荐，可直观调速率）：

```powershell
cd source
python sensor_simulator_gui.py
```

在界面中选择 `..\data\pressure_timestep.csv`，起始时间步填 `88000`，发送速度 `3000` 行/秒，每批行数 `50`，点击“检查文件”后“开始输出”。

或使用命令行版本：

```powershell
cd source
python simulate_sensor.py ..\data\pressure_timestep.csv --start-step 88000 --rate 3000 --batch-size 50
```

### 观察结果

- **在线监测**页：上图为周向 RMS / 单探针压力时域波形，下图为各探针的风险分数曲线与阈值虚线；超限探针曲线变红。
- **小波分析**页：所选探针的连续小波（CWT）功率尺度图，横轴物理时间、纵轴频率，供人工辅助观察，**不参与报警**。
- 满足判据的连续触发窗口数后系统红灯点亮，右侧「事件记录」逐条记录探针报警与动作机构状态。
- 结束时点击「停止监听」，再点击「导出监听结果」，保存 CSV 与同名 JSON 上下文（有动作历史时另生成 `_actuation.csv` 动作指令日志）。

---

## 2. 文件清单

### 顶层

| 路径 | 作用 |
| --- | --- |
| `README.md` | 本文件：项目说明、文件清单、运行步骤。 |
| `使用教程.md` | 面向演示与实验接入的操作教程（含打包版 EXE 的使用方式）。 |
| `项目历史.md` | 源码恢复过程、各版本演进与验证记录。 |
| `data/` | 演示数据（见下）。 |

### `data/` 演示数据

| 文件 | 作用 |
| --- | --- |
| `pressure_u1_initialization.csv` | **初始化标定数据**。5 个周向探针（Monitor Point 1/3/9/5/7）的壁面压力，单位 Pa，首列累计时间步从 60000 开始，整段为已知正常工况。 |
| `pressure_timestep.csv` | **回放数据**。同一测量链路、同样 5 探针，包含从正常逐渐发展到失速的完整过程，用于演示失速识别。 |

两个文件的格式相同：第一列是累计时间步，其余数值列自动识别为探针，列名中的 `(Pressure) [Pa]` 会被解析为物理量类型与单位。程序没有写死探针数量，但计算、内存与绘图成本随探针数近似线性增长。

### `source/` 入口脚本

| 文件 | 作用 | 用法 |
| --- | --- | --- |
| `app.py` | **主程序入口**。创建 `QApplication`、加载深色主题 `assets/theme.qss`、安装全局异常钩子并打开 `MainWindow`。 | `python app.py`；自检模式 `python app.py --smoke-test ..\data\pressure_timestep.csv` |
| `sensor_simulator_gui.py` | **图形化传感器模拟器**。选择 CSV、设定响应输出间隔/目标地址/发送速度/每批行数/起止时间步，后台线程发送，带进度、日志与中断。 | `python sensor_simulator_gui.py` |
| `simulate_sensor.py` | **命令行传感器模拟器**。连接 `127.0.0.1:8765`，先发 `meta` 握手再分批发 `samples`。 | `python simulate_sensor.py <csv> --start-step 88000 --rate 3000 --batch-size 50` |
| `validate_baseline.py` | **离线假报警校验**。用一段正常数据建立基线，再让另一段连续工况数据离线通过同一判据，统计失速发生前的报警窗口数；退出码 `2` 表示存在失速前误报。 | `python validate_baseline.py baseline.csv monitor.csv --method autocorrelation [--known-stall-step 105000]` |
| `run_tests.py` | **一键验证**。依次运行单元测试、主程序冒烟自检、模拟器冒烟自检。 | `python run_tests.py` |
| `build.ps1` | **PyInstaller 打包脚本**。分别打包主程序与模拟器为 `--onedir --windowed`，并把数据与启动脚本复制进产物目录。 | `.\build.ps1 [-BuildTag my_version]`，产物在 `source\dist\<标签>\` |
| `run_simulator.ps1` | 启动**已打包**的 `SensorSimulator.exe` 的便捷脚本（源码运行无需此文件）。 | `.\run_simulator.ps1 -Csv ..\data\pressure_timestep.csv -StartStep 88000` |
| `requirements.txt` | 依赖：numpy、pandas、scipy、PyWavelets、PySide6、pyqtgraph、pyinstaller。 | `python -m pip install -r requirements.txt` |
| `README.md` | 源码工程的开发、验证与打包说明，以及设计约束与主动控制接口文档。 | — |

### `source/src/` 数值核心

| 文件 | 作用 |
| --- | --- |
| `config.py` | 资源与配置定位：`resource_path()` 兼容源码态与 PyInstaller 冻结态，`load_defaults()` 读取 `config/defaults.json`。 |
| `models.py` | 核心数据结构：`DataSourceSettings`（采样间隔与周向方向，含校验）、`ProbeDataset`（时间步、物理时间、探针名、角度、数值矩阵、单位、质量信息与 `sample_rate_hz` 等属性）、`SystemAnalysisResult`（离线分析结果）。 |
| `io_csv.py` | CSV 解析入口：编码嗅探（utf-8-sig/utf-8/gb18030/latin-1）、表头与分隔符识别、按累计时间步去重取均值、缺失值统计、物理时间换算（`first_row_zero` 或 `step_origin`）、探针名简化、单位提取与等角度生成。 |
| `preprocessing.py` | 信号预处理：`interpolate_missing()`、`apply_filter()`（零相位 Butterworth，离线用）、`causal_rolling_rms()`（只依赖历史样本，在线用）、`OnlineSignalFilter`（带状态的流式滤波器，可复位）、`normalize_rms()`、`circumferential_amplitude()`。 |
| `online_engine.py` | **在线因果预警引擎**。`OnlineWarningEngine` 用定长环形缓冲增量计算：标定阶段累计指定窗数后冻结每探针基线；打分阶段输出 `OnlineSnapshot`（窗口时间、特征、风险分数、探针/系统报警、标定进度）。含逐探针滞回状态机（连续触发/连续回落）与 K-of-N 系统确认，并支持基线档 `calibration_profile()` 的导出与严格校验载入。 |
| `analysis_service.py` | 离线分析编排与判据派发：`_single_feature()` 按方法名分派到三种特征函数（在线引擎复用同一分派），`analyze_signals()` 串起特征 → 稳健打分 → 滞回 → K-of-N，`METHOD_LABELS` 提供中文判据名。 |
| `warning_engine.py` | 稳健基线工具与通用状态机：`fit_robust_baseline()`（中位数 + 1.4826×MAD）、`score_with_baseline()`（支持高/低/双侧方向）、`apply_hysteresis()`、`aggregate_probe_alarms()`。 |
| `stall_tracking.py` | **失速团周向定位**。优先用相邻探针报警上升沿的时间差拟合角速度并外推位置；退化时用短时压力周向对比度圆均值定位；再退化为风险值周向重心。 |
| `actuation.py` | **16 通道主动控制基础件**。`ActuatorController` 默认全部通道未使能（安全默认），报警时选择距目标角最近的两个已使能通道下发命令；`PulseScheduler` 按物理时间管理重复脉冲；可注册每通道或整帧回调，并维护启动计数与动作历史。 |
| `streaming.py` | 流式数据源抽象（`SampleBatch`、`CsvReplaySource`）。**当前未被引用**，为后续回放接口预留。 |
| `workers.py` | Qt 后台任务工具（`FunctionWorker` + 信号）。**当前未被引用**，为通用异步任务预留。 |

### `source/src/algorithms/` 判据算法

| 文件 | 作用 |
| --- | --- |
| `autocorrelation.py` | 时序自相关判据：逐窗逐探针标准化后，在 `[min_lag, max_lag]` 内取归一自相关峰值作为特征（失速先兆通常表现为该峰值的系统性变化）。 |
| `entropy.py` | 排列熵判据：归一化排列熵（0~1），熵升高代表信号趋于随机。 |
| `wavelet.py` | 小波细节能量比判据：`pywt.wavedec` 多层分解，取指定细节层能量占总能量的比例。 |
| `common.py` | 公共窗口工具：`window_indices()`、`window_times()`、`standardize_window()`（有效点比例不足则丢弃该窗）。 |
| `__init__.py` | 导出三个特征函数。 |

### `source/src/ui/` 界面

| 文件 | 作用 |
| --- | --- |
| `main_window.py` | **主窗口**（约 1000 行）。三列布局：左侧参数面板（数据与监听 / 实时滤波 / 机器判据 / 显示），中间 `QTabWidget`（“在线监测”页含时域波形与风险分数图，“小波分析”页含 CWT 尺度图），右侧状态面板（报警灯、事件记录、TCP 状态、16 通道主动控制）。负责初始化标定、TCP 监听与握手校验、实时判定、绘图节流与结果导出。 |
| `data_source_dialog.py` | 数据映射对话框：设定响应输出间隔、首样本是否归零、首探针角度与周向方向，并弹出数据集确认框（样本数、探针数、周向间隔、物理时长、单位）。 |
| `actuator_panel.py` | 16 通道主动控制面板：`ActuatorRingWidget` 自绘周向环形机构图（1 号位于正上方 0°，顺时针增编号），支持点击选通道、使能、待机偏角与喷气持续时间；“控制 / 动作记录”两个子页显示喷口方向与各通道累计启动次数。 |

### `source/tests/` 与其它

| 文件 | 作用 |
| --- | --- |
| `tests/test_core.py` | 核心单元测试：CSV 契约与时间原点、周向角度生成、探针数动态识别、排列熵与自相关特征形状、滞回 + K-of-N 状态机、在线引擎的整段/分块结果一致性、因果 RMS、基于探针报警时差的失速定位与退化策略、主动控制选通道与计数语义、脉冲调度。 |
| `assets/theme.qss` | 全局深色主题样式表，由主程序与模拟器加载。 |
| `config/defaults.json` | 默认配置，见下节。 |

---

## 3. 配置说明（`source/config/defaults.json`）

| 分组 | 关键项 | 含义 |
| --- | --- | --- |
| `data_source` | `output_interval_s` | 相邻数据行的物理时间间隔（秒）。演示数据为 `5.76928006151677e-6`，即采样率约 173.3 kHz。 |
| | `step_origin` / `first_row_zero` | 首个样本的物理时间原点；`first_row_zero=true` 时首行固定显示为 `0 s`。 |
| | `start_angle_deg` / `angle_direction` | 首探针周向角度与编号方向（±1）。 |
| `rms` | `half_window` / `downsample` / `amplitude_scale_deg` | 显示用 RMS 的半窗口、绘图抽点步长、RMS 到角度振幅的视觉映射。 |
| `online` | `calibration_windows` | 标定所需的分析窗数（默认 153）。 |
| | `timer_interval_ms` | 界面重绘节流（默认 66 ms，即约 15 FPS）；数据接收与绘图解耦，机器判据仍逐窗计算。 |
| `filter` | `mode` / `order` | 实时因果滤波开关与阶数。 |
| `analysis` | `machine_method` | 机器判据，可选 `autocorrelation` / `entropy` / `wavelet`。 |
| | `window_size` / `overlap_percent` | 分析窗长度与重叠率（默认 1024 点、50%）。 |
| | `threshold` | 风险分数阈值，单位是稳健标准差（默认 3.0）。 |
| | `trigger_windows` / `release_windows` / `release_ratio` | 连续触发窗数、连续解除窗数、解除阈值比例（默认 3 / 5 / 0.7）。 |
| | `minimum_probes` | K-of-N 的 K，需要同时报警的探针数（默认 2）。 |
| | `wavelet` / `wavelet_level` / `detail_level` | 小波判据的小波基、分解层数与关注的细节层。 |
| | `entropy_order` / `entropy_delay` | 排列熵的嵌入维数与延迟。 |
| | `min_lag` / `max_lag` | 自相关特征搜索的滞后范围。 |

说明：`filter.low_cutoff_fraction_fs`、`filter.high_cutoff_fraction_nyquist`、`online.replay_speed` 为历史预留字段，当前代码未读取。

---

## 4. 测试与校验

```powershell
cd source
python run_tests.py                                   # 单元测试 + 主程序/模拟器冒烟自检
python -m unittest discover -s tests -v               # 仅单元测试
python app.py --smoke-test ..\data\pressure_timestep.csv          # 主程序无头自检
python sensor_simulator_gui.py --smoke-test ..\data\pressure_timestep.csv   # 模拟器无头自检
```

判断“逐渐关阀时是否误报”（正式实验前建议执行）：

```powershell
python validate_baseline.py ..\data\pressure_u1_initialization.csv ..\data\pressure_timestep.csv --method autocorrelation
```

用已知真正失速的累计时间步可区分失速前误报与失速后报警：

```powershell
python validate_baseline.py ..\data\pressure_u1_initialization.csv ..\data\pressure_timestep.csv --method autocorrelation --known-stall-step 105000
```

---

## 5. 常用的 `simulate_sensor.py` 参数

| 参数 | 说明 |
| --- | --- |
| `csv` | 要发送的数据文件（默认找同目录及附近的 `pressure_timestep.csv`）。 |
| `--start-step` / `--end-step` | 按累计时间步截取发送区间。`--start-step 88000` 对应该数据中压力开始明显下降的区段。 |
| `--max-rows` | 最多发送的行数；不设则发到文件结束。 |
| `--rate` | 发送速率（行/秒），`0` 表示不限速。 |
| `--batch-size` | 每条 TCP 消息包含的行数。 |

关于发送速率：`--batch-size 1` 时，速率受 Windows 定时器精度（约 15.6 ms）限制，单独把 `--rate` 提到几百并不准确。需要快速回放时请同时增大批量，例如 `--rate 3000 --batch-size 50`（每批睡眠约 17 ms，节奏稳定）。

---

## 6. 设计约束与已知限制

- 初始化数据必须是**同一测量链路**下的已知正常工况，且探针名称、顺序、响应输出间隔与在线阶段一致（握手时程序会校验）。
- 红绿灯只跟随当前选择的机器判据（时序自相关 / 排列熵 / 小波能量）；“小波分析”页的 CWT 尺度图仅供人工观察。
- 系统报警采用 K-of-N、连续触发窗口与连续解除窗口，避免单点毛刺。
- 在线首个样本的物理时间固定为 `0 s`。
- 默认窗口重叠率 50%，界面约 15 FPS 刷新；RMS 与判据图只处理当前可见时间段，小波只在对应页处于可见状态时更新；机器判据仍逐窗计算，不因绘图限帧而漏判。
- `3.0` 阈值的含义是“偏离稳健基线 3 个标准差”，属于统计约定，**不具备量纲安全裕度含义**，不能直接用作安全联锁定值。
- 更换阀门开度、转速或流量后，特征分布会变化，应重新标定并进行跨工况假报警验证。
- 16 通道主动控制目前只提供软件回调与动作日志，尚未绑定 PLC、串口或现场总线；接入真实硬件前必须确定协议、刷新率、喷气持续时间与失效安全状态，并完成独立联锁验证。
