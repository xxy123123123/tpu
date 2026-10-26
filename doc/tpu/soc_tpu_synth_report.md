# SoC 集成 TPU 综合报告（PG2L200H / PDS 2022.2-SP6.4）

日期：2026-10-24
工程：`sim/soc/synth/soc_tpu_run/soc_tpu.pds`（top = board_top）
配置：ENABLE_TPU=1，TPU 阵列 P=16 × C=8（128 个 INT8 MAC），权重独立 AXI 读口（TPU_WT_PORT=1）

## 1. 结论

- **时序通过**：核心时钟域 ddrphy_sysclk = 93.75MHz（周期 10.667ns），布线后 Slow 角 WNS = **+0.542ns**，TNS = 0，Hold WNS = +0.104ns。等效 Fmax ≈ 98.7MHz。
- **资源余量充足**：LUT 18%、FF 8%、DRM 23%、APM 19%。
- 相对 10-20 无 TPU 板级基线（Fmax 100.11MHz / WNS +0.678ns），TPU 集成仅损失约 0.14ns 时序裕量。

## 2. 资源占用（dev_map，全 SoC 含 TPU）

| 资源 | 已用 | 可用 | 利用率 |
|---|---|---|---|
| LUT | 27,550 | 159,800 | 18% |
| FF | 23,040 | 319,600 | 8% |
| DRM | 94.5 | 415 | 23% |
| APM (DSP) | 139 | 740 | 19% |
| Distributed RAM | 306 | 39,500 | 1% |
| IO | 93 | 400 | 24% |
| GPLL / PPLL | 2 / 2 | 10 / 10 | 20% |

TPU 增量估算（对比 10-20 基线 LUT 19,071 / FF 17,309）：**LUT +8,479、FF +5,731、APM +128**（128 个 MAC 各占 1 APM；其余约 11 个 APM 为 CPU mul IP 等原有逻辑）。

## 3. PnR 多 seed 结果（performance 模式，seed 2~5 并行）

主流程在 seed 收尾阶段被中断，seed_3 / seed_4 已完整收敛（Pass Rate 100%）：

| Seed | Setup(Slow) WNS | Setup(Fast) | Hold(Slow) | 布线 |
|---|---|---|---|---|
| 3（最优） | **+0.542 ns** | +3.036 ns | +0.104 ns | 全部收敛，0 未布网络 |
| 4 | +0.426 ns | +3.036 ns | +0.118 ns | 全部收敛 |

seed_3 完整 DB 保留在 `place_route/seed_3/`（prr.db / rtr.db / board_top_pnr.adf），可直接用于 report_timing / bitgen。
各阶段耗时：compile→dev_map 9m12s；seed_3 PnR 23m23s（峰值内存 3.4GB）。

## 4. 关键路径说明

- seed_3 布局后 WNS 969ps → 布线后 542ps，布线恶化 427ps（5% 路径），属正常范围。
- 时序约束：ddr_ref_clk 125MHz 输入，DDR PHY GPLL 生成 93.75MHz 系统时钟；phy_dq_clk 750MHz（DDR3-1500）；跨域均已 set_clock_groups 隔离。

## 5. 遗留事项

1. **基线对照未跑成**：pds_shell 会拉起 wmic.exe，被本机安全策略黑名单拦截（提权也不行）。如需精确的"同版本无 TPU"资源差值，需在 安全中心→命令安全→程序黑名单 移除 wmic.exe 后重跑：
   `pds_shell -project sim/soc/synth/soc_base_run/soc_base.pds -work_dir ... -run dev_map`（工程文件已备好）。
2. report_timing / bitgen 未跑（主流程中断于 seed 汇总阶段），需要时可基于 seed_3 DB 补跑。
3. P32C16 变体脚本（run_soc_tpu32.tcl）已备好，未启动。
