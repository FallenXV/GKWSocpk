# GKWSocpk

[English](README.md) | 简体中文

将 [socpk.com](https://socpk.com) 的 CPU、GPU 和电池测试结果抓取为 CSV
快照，并在本地网页仪表盘中进行对比。

![仪表盘](docs/images/dashboard.png)

这是个人项目，与 socpk.com 无关。使用时请遵守该网站的服务条款。

## 安装

需要 Python 3.13。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          # bash/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

如需精确复现环境，`requirements.lock.txt` 锁定了一套已验证可用的依赖版本。

## 1. 采集数据

```powershell
python Battery\battery_parser.py --auto-soc
python "Performance Benchmark\cpu_curve_parser.py" --benchmark all
python "Performance Benchmark\gpu_curve_parser.py"
python "Performance Benchmark\laptop_gpu_curve_parser.py"
```

快照写入 `snapshots/` 目录：

| 文件 | 内容 |
| --- | --- |
| `cpu_gb6_curves.csv` | Geekbench 6 多核 |
| `cpu_gb7_curves.csv` | Geekbench 7 多核 |
| `cpu_spec2026_int_curves.csv` | SPEC CPU 2026 整数，按核心 |
| `cpu_spec2026_fp_curves.csv` | SPEC CPU 2026 浮点，按核心 |
| `gpu_snl_curves.csv` | 手机 GPU，3DMark Steel Nomad Light |
| `laptop_gpu_curves.csv` | 笔记本 GPU，3DMark Time Spy |
| `battery_results.csv` | 续航测试 5.0 的续航时间、容量和功耗 |

已有文件永远不会被覆盖。如果文件名已被占用，新快照会加上时间戳后缀，
仪表盘会加载最新的一份。

常用选项：

- **CPU：** `--benchmark GB6|GB7|SPEC2026_INT|SPEC2026_FP|all`、`--cpus "A19 Pro"`、
  `--core-group super|large|medium|small`
- **GPU：** `--gpus` 用于限定要抓取的 GPU
- **电池：** `--auto-soc` 会在 GSMArena 上查询每款手机的处理器，并将结果缓存到
  `.gsm_cache/`。`--spec-offline` 只使用该缓存，`--spec 'Brand|Model=url'`
  可手动修正单款手机。

任意脚本加 `--help` 可查看完整选项列表。

## 2. 打开仪表盘

```powershell
python socpk_web.py
```

该命令会在浏览器中打开仪表盘。它只运行在 `127.0.0.1` 上，不会上传任何数据。
按 Ctrl-C 停止。

每个标签页对应一个数据集：GB6 MULTI、GB7 MULTI、SPEC26 INT、SPEC26 FP、GPU、
LAPTOP GPU 和 BATTERY。在 **Compare** 中选择图表，然后在左侧列表中选择配置项
（profile）。单击可切换选中状态，Shift+单击可选择一个范围，也可以使用
**Top 5**、**All shown** 和 **Clear**。每个图表右侧都有排名面板，鼠标悬停在
数据点上可查看详情。**Export chart** 可将图表保存为 PNG 或 SVG。

每个图表中的青色小三角形标出结果更优的角落，图表下方的一行说明会解释该角落的含义
（例如左上角：功耗更低、分数更高）。

**Efficiency reference** 开关用于绘制下文介绍的虚线参考线。图表下方的说明
会解释当前显示的参考线。

## 性能图表（CPU 和 GPU）

六个 CPU 和 GPU 标签页都提供相同的三种图表。每条曲线代表一颗芯片，
在 SPEC 标签页中则代表一个核心。

| 性能曲线 | 能效曲线 | 能效 vs 分数 |
| --- | --- | --- |
| ![](docs/images/cpu-performance-curve.png) | ![](docs/images/cpu-efficiency-curve.png) | ![](docs/images/cpu-efficiency-vs-score.png) |
| 分数 vs 功耗（W） | 每瓦分数 vs 功耗（W） | 每瓦分数 vs 分数 |
| 按峰值分数排名 | 按峰值每瓦分数排名 | 按峰值每瓦分数排名 |

- **性能曲线（Performance curve）：** 芯片在各功耗水平下能提供多少性能。
  越高、越靠左越好。从原点出发的虚线射线表示恒定的每瓦分数。粗射线穿过所选
  数据中能效最高的点，浅色射线分别表示该能效的 50%、25% 和 12.5%。
- **能效曲线（Efficiency curve）：** 能效随功耗上升如何变化。通常在低功耗时
  达到峰值，随着芯片负载加重而下降。
- **能效 vs 分数（Efficiency vs score）：** 达到某一分数所需的能效。这是在同等
  性能下比较芯片最直接的方式。

数据点均为 SoCPK 公布的数值，其中部分是上游拟合所得。Apple 芯片每颗只公布
少量数据点，因此曲线较为稀疏。

在 SPEC 标签页中，**Filter cores** 可按核心分组（Super、Large、Medium、Small）
或核心名称筛选列表。

## 电池图表

每款手机都有一个 SoCPK 续航测试 5.0 的续航时间，以及一个取自电池标称信息的
电池容量（Wh）。图表由此推导出两个数值：

- **平均功耗（W）** = 容量（Wh）÷ 续航时间（h）。越低越好。
- **能效（min/Wh）** = 续航时间（分钟）÷ 容量（Wh）。越高越好。

这些数值反映的是整机表现，包括屏幕、基带和软件，而不仅是芯片本身。

在设备图表中，空心菱形表示极客湾（Geekerwan）实测的可用容量（如有），并用
虚线与 SoCPK 数据点相连。这些实测值仅供参考，不影响排名。可通过
**Geekerwan measured capacity** 关闭显示。

### 续航时间 vs 容量

![续航时间 vs 容量](docs/images/battery-runtime-vs-capacity.png)

该图表显示续航时间（小时）与电池容量的关系，并按续航时间对手机排名。虚线表示
在所选手机中能效最高者的功耗下，各电池容量可达到的续航时间。越接近该线的手机，
电池利用越好。

### 能效

![能效](docs/images/battery-energy-efficiency.png)

该图表布局相同，但续航时间以分钟为单位，并按 **每 Wh 分钟数** 对手机排名。
这一排名将能效与电池容量区分开，因此小电池手机也可能排在大电池手机之前。

### 平均功耗

![平均功耗](docs/images/battery-average-power-draw.png)

该图表显示平均功耗与电池容量的关系，并按功耗从低到高对手机排名。虚线连接
带圆圈标记的手机：对其中每一款而言，所选手机中没有另一款同时拥有更大的电池
和更低的功耗。

### 处理器平均值

如果快照是使用 `--auto-soc` 采集的，左侧面板会多出一个 **Processors** 标签页。
每个处理器的数值是使用该处理器的手机的简单平均，每款手机只计一次。无法可靠
识别芯片的手机不计入。

| SoC 平均功耗 | SoC 平均能效 |
| --- | --- |
| ![](docs/images/battery-soc-average-power.png) | ![](docs/images/battery-soc-average-efficiency.png) |
| 平均 W vs 平均电池 Wh，按 W 从低到高排名 | 平均 min/Wh vs 平均 W，按 min/Wh 从高到低排名 |

在设备图表中，**Overlay average lines** 会为每个选中的处理器绘制一条带标签的线。

## 分析实验室

图表下方的 **Analysis lab** 面板基于所选配置项进行分析。

- **CPU 和 GPU 标签页：**
  - 指定功耗下的分数，以及达到目标分数所需的功耗
  - 达到峰值性能 80–100% 所需的功耗
  - 帕累托前沿
  - 芯片代际之间的基准对比
- **电池标签页：**
  - 按处理器分组的手机功耗，支持品牌和屏幕筛选
  - 续航对比，拆分为电池容量影响和功耗影响

公布数据点之间的数值采用线性插值，不会在范围之外外推。单击列标题可对表格排序，
使用 **Export analysis CSV** 可保存表格。

`battery_metadata.json` 保存经过审核的手机元数据修正，例如芯片组、屏幕尺寸和
刷新率。这些修正在加载数据时应用，不会改变续航时间或容量。

## 仪表盘选项

```powershell
python socpk_web.py --dataset Battery              # 打开指定标签页
python socpk_web.py --csv snapshots\gpu_snl_curves.csv
python socpk_web.py --port 0 --no-browser          # 任意空闲端口，不打开浏览器
python socpk_web.py --https                        # 适用于开启 HTTPS-Only 的 Safari
python socpk_web.py --browser chrome
```

- **Safari：** 开启 HTTPS-Only 后，Safari 会拒绝访问普通的 `http://localhost`。
  请使用 `--https`（会生成本地自签名证书），或改用其他浏览器打开仪表盘。
- **Tk 备用界面：** `socpk_gui.py` 是速度较慢的桌面备用程序，提供相同的数据集和
  图表，适用于没有可用浏览器的机器。它不包含 Analysis lab。
- **键盘快捷键：** `[` 和 `]` 切换标签页，`/` 聚焦搜索框。

## 其他脚本

- `Performance Benchmark\curve_analysis.py --input <csv> --save` 绘制每颗芯片的
  功耗、分数和能效统计图，并写入 `curve_summary.csv`。
- `analysis/battery_blog_audit.py` 可离线复现电池分析文章中的数据。

## 检查改动

```sh
python tests/smoke.py
```

该命令会编译所有源文件，在进程内启动仪表盘，并检查其页面和 API 是否正常响应。
成功时输出 `SMOKE OK`。
