# SoC 与 TPU 架构说明

> 本文档说明 TPU 卷积加速器如何挂进 RISC-V SoC、数据如何流动、如何综合与仿真复现。
> 面向接手本项目的硬件/集成同事。

---

## 1. 总体结构

```
                        ┌──────────────────────────────────────────────┐
                        │                 board_top                    │
                        │   (盘古MINI PG2L200H 顶层，含 DDR3 / 摄像头)  │
                        └───────────────────┬──────────────────────────┘
                                            │ ENABLE_TPU / TPU_P / TPU_C
                        ┌───────────────────▼──────────────────────────┐
                        │                soc_ddr3_top                  │
                        │  ┌────────┐        ┌─────────────────────┐   │
                        │  │ RISC-V │  MMIO  │  g_tpu (generate)   │   │
                        │  │  CPU   │───────▶│  ┌───────────────┐  │   │
                        │  │ RV32I  │ 0x1000 │  │  tpu_mmio     │  │   │
                        │  └────────┘ _5000  │  └──────┬────────┘  │   │
                        │        │           │         │ 控制       │   │
                        │        │           │  ┌──────▼────────┐  │   │
                        │        │           │  │  tpu_dma      │  │   │
                        │        │           │  │ (激活/权重/写) │  │   │
                        │        │           │  └──────┬────────┘  │   │
                        │        │           │         │ 数据       │   │
                        │        │           │  ┌──────▼────────┐  │   │
                        │        │           │  │tpu_conv_engine│  │   │
                        │        │           │  │ P16×C8 INT8   │  │   │
                        │        │           │  └───────────────┘  │   │
                        │  ┌─────▼──────┐    └──────────┬──────────┘   │
                        │  │  DDR3 控制器│◀────AXI 仲裁────┘             │
                        │  │  (1GB)     │                              │
                        │  └────────────┘                              │
                        └──────────────────────────────────────────────┘
```

三个核心模块（均在 `myriscv/`）：

| 模块 | 职责 |
|---|---|
| `tpu_conv_engine.v` | INT8 卷积引擎，P16×C8=128 MAC/cycle，纯计算 |
| `tpu_dma.v` | DMA：从 DDR 读激活/权重、写回结果，AXI 主口 |
| `tpu_mmio.v` | MMIO 从设备：CPU 配寄存器、触发计算、读状态 |

---

## 2. TPU 挂载方式

### 2.1 集成位置

`myriscv/soc_ddr3_top.v` 内 `generate if(TPU_ACTIVE) begin : g_tpu` 块完成实例化。

顶层 `myriscv/board_top.v` 通过参数使能：

```verilog
parameter integer ENABLE_TPU = 1,   // 1 = 启用 TPU
parameter integer TPU_P = 16,       // 输出通道并行度（冻结）
parameter integer TPU_C = 8;        // 输入通道并行度（冻结）
```

### 2.2 关键参数（`soc_ddr3_top.v`）

| 参数 | 值 | 说明 |
|---|---|---|
| `TPU_P` / `TPU_C` | 16 / 8 | 阵列并行度，**已冻结** |
| `TPU_MAX_W` | 256 | 输入宽度上限 |
| `TPU_NCBMAX` | 8 | 行缓冲通道块维度（cin ≤ 64） |
| `TPU_WT_PORT` | 1 | 权重独立 AXI 读口（实测快 14%，默认开） |

### 2.3 MMIO 窗口

- **基址**：`0x1000_5000`，256 字节，word 对齐
- 由 `data_bus_interconnect` 译码，CPU 通过 load/store 访问
- 具体寄存器偏移见 `TPU权重文件格式规范.md` 第 8 节（R_WT_AB=0x58、R_DMA_AB=0x40 等）

### 2.4 数据通路（DMA）

`tpu_dma` 作为 AXI 主设备，挂在 SoC 的 AXI 仲裁链上：

- **读口 1（激活）**：从 DDR 读输入特征图
- **读口 2（权重）**：从 DDR 读权重（`TPU_WT_PORT=1` 时独立）
- **写口**：把卷积结果写回 DDR

仲裁顺序：`cnn -> tpu_in -> [tpu 内部仲裁] -> AXI -> DDR`。

---

## 3. 与旧 CNN 引擎的关系

项目里同时存在两套 AI 引擎，由参数切换：

| 引擎 | 文件 | 参数 |
|---|---|---|
| 旧 CNN | `cnn_digits_engine.v` 等 6 个 | `ENABLE_CNN`（board_top 内硬编码为 1） |
| 新 TPU | `tpu_conv_engine.v` 等 3 个 | `ENABLE_TPU` |

> **当前主力是 TPU**。旧 CNN 保留用于对照/兼容，非本项目主线。
> 两者在 `soc_ddr3_top` 内互不冲突，共用同一 AXI 仲裁链。

---

## 4. 一次推理的数据流（软硬协同）

1. **CPU 准备数据**：把激活图、权重、偏置写入 DDR（C 程序完成）
2. **CPU 配置寄存器**：写 `tpu_mmio` 的 H/W/cin/cout/stride/pad/ksize/relu/shift
3. **CPU 触发**：写 DMA 基址 + 启动位
4. **DMA 搬数**：从 DDR 读激活（喂 npg 遍）+ 权重 → 引擎
5. **引擎卷积**：128 MAC/cycle 并行计算，逐行滑行复用
6. **DMA 回写**：结果写回 DDR
7. **CPU 读回**：读结果，做 GAP/argmax 等后处理（CPU 完成）

完整可运行示例：`tests/tpu_conv/main.c`（SoC 级协同仿真已验证 PASS）。

---

## 5. 综合复现（上板）

### 5.1 工具链前提

- **紫光同创 PDS 2022.2-SP6.4**
- 板卡：盘古 MINI（PG2L200H-6FBB676，Logos2）
- DDR3 IP（`ipcore/ddr3/`，需对应 IP 授权）

### 5.2 综合工程

SoC 级综合工程入口：

```
sim/soc/synth/soc_tpu_run/soc_tpu.pds      # top = board_top，ENABLE_TPU=1
sim/soc/synth/run_soc_tpu.tcl              # 一键综合脚本
```

> 注意：`.pds` / `.tcl` 内是**绝对路径 `D:/riscv/RISCV/...`**。
> 接手者 clone 后需把路径改成自己的本地路径（或保持相同目录结构）。

### 5.3 已验证结果

| 项 | 值 | 来源 |
|---|---|---|
| 时序 | 布线后 Slow WNS +0.542ns，等效 Fmax ≈ 98.7MHz | `soc_tpu_synth_report.md` |
| 系统主频 | 93.75 MHz | DDR PHY GPLL |
| 资源 | LUT 18% / FF 8% / DRM 23% / APM 19% | 同上 |

---

## 6. 仿真复现

### 6.1 引擎级回归（23 用例 bit-exact）

```bash
python sim/tpu/run_regress.py
```

> 需要 Python + numpy + iverilog。Windows 下注意 iverilog 依赖 MSYS2 DLL，
> 需把 `D:/vsc/msys64/ucrt64/bin` 加进 PATH。

### 6.2 SoC 级协同仿真

```bash
python sim/tpu_soc/run.py
```

真实 CPU 从 DDR 启动 `tests/tpu_conv`，软硬件协同，打印 `TPU_SOC_PASS`。
单次约 7~8 分钟（340 万周期）。

---

## 7. 文件清单（TPU 相关，不含 CPU 逻辑）

| 文件 | 说明 |
|---|---|
| `myriscv/tpu_conv_engine.v` | 卷积引擎（核心） |
| `myriscv/tpu_dma.v` | DMA 搬运 |
| `myriscv/tpu_mmio.v` | MMIO 寄存器接口 |
| `myriscv/soc_ddr3_top.v` | SoC 集成层（含 g_tpu 例化） |
| `myriscv/board_top.v` | 顶层（ENABLE_TPU 参数） |
| `sim/tpu/` | 引擎级回归（23 用例） |
| `sim/tpu_soc/` | SoC 级协同仿真 |
| `sim/tpu/synth/` | 引擎单独 PnR 评估 |
| `sim/soc/synth/soc_tpu_run/` | SoC 综合工程（出位流） |
| `tests/tpu_conv/` | C 程序（软硬协同测试） |
| `doc/tpu/` | 本说明 + 权重格式文档 + 综合报告 |

---

*本文档依据 `myriscv/tpu_*.v`、`soc_ddr3_top.v`、`board_top.v` 逐项核对。*
