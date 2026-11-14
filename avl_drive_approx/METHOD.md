# AVL-DRIVE 风格开源近似算法说明

> 本仓库实现的是**方法学对标用的开源近似**，不是 AVL-DRIVE 闭源算法的逆向还原。  
> 传递函数节点、权重可按车级/品牌 DNA 自行标定。

## 统一流水线

```text
信号 → Trigger 工况识别 → 物理特征提取 → 分段线性传递函数 → 局部 DR → 加权合成 overall DR(1~10)
```

## 1. Tip-in（纵向驾驶性）

**触发**：踏板上升速率与阶跃幅值同时超过阈值。

**特征**

| 特征 | 定义 |
|---|---|
| response_delay | 踏板起跳 → \(a_x\) 超过基线+0.15 m/s² 的时间 |
| rise_rate | 起跳后约 0.4 s 的平均加速度上升率 |
| hesitation | 0.3–1.2 s 窗内加速度峰-谷跌落 |
| oscillation | 1.5 s 内 \(\int |\dot a_x|\,dt\)（shuffle/thump 代理量） |

**评分**：各特征经 `piecewise_linear_score` 映射为 1–10，默认权重 delay 0.30 / rise 0.20 / hes 0.20 / osc 0.30。

## 2. 回收↔机械制动交接（VIORE 风格）

**触发**：持续减速需求（\(a_x\) 或 brake_req）。

**特征**

| 特征 | 定义 |
|---|---|
| regen_utilization | \(\sum |T_{motor}| / \sum(|T_{motor}| + T_{hyd})\) |
| handover_bump | 液压首次明显介入附近的 \(\max|\dot a_x|\) |
| mode_difference | 两回收模式平均减速度之差（希望可感知但不过激） |
| reproducibility | 多次试验平均 \(a_x\) 的标准差 |

## 3. ACC Cut-in（ADAS）

**触发**：前车距离在短窗内突降（规则型 cut-in）。

**特征**

| 特征 | 定义 |
|---|---|
| response_delay | cut-in 时刻 → ego \(a_x\) 首次 ≤ -0.4 m/s² |
| brake_harshness | 事件窗内最负 \(a_x\) |
| jerk | 事件窗内最负 \(\dot a_x\) |
| min_time_headway | \(\min(s/v)\) |
| speed_undershoot | 相对事件前车速的超调跌落 |

## 4. 垂向舒适（Ride）

**触发**：\(|a_z|\) 超过冲击阈值；否则按连续路面全窗评价。

**特征**

| 特征 | 定义 |
|---|---|
| peak_to_peak | 窗内 \(a_z\) 峰峰值 |
| rms | \(a_z\) RMS |
| bounce_band | 1–3 Hz 带通 RMS（车身起伏代理） |
| wheel_hop_band | 8–15 Hz 带通 RMS（轮跳代理） |
| longitudinal_coupling | \(\mathrm{mean}(|a_x||a_z|)\)（可选） |

生产环境建议把 FFT 带通替换为 ISO 2631 计权滤波，并用座椅导轨标定。

## 快速运行

```bash
python -m avl_drive_approx.examples.run_demo
python -m pytest tests -q
```
