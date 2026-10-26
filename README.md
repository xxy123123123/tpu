# RISC-V SoC + TPU 卷积加速引擎（交付仓库）

> 本仓库收录 **SoC 与 TPU 交接说明** 与 **TPU 相关源码**，供接手者/协作者复现、验证与继续开发。
> 内容从主开发仓库 `D:\riscv\RISCV` 精选导出，版本口径以引擎文件头标注 **v9** 为准。

---

## 1. 这是什么

盘古 MINI 开发板（紫光同创 Logos2 PG2L200H）上的 RISC-V CPU SoC 中，挂载了一块 **INT8 卷积加速引擎（TPU）**：

| 模块 | 说明 |
|---|---|
| `tpu_conv_engine.v` | INT8 卷积引擎，P16×C8 = 128 MAC/cycle，三级流水 + 5-bank 内容寻址行缓冲（**v9**） |
| `tpu_dma.v` | DMA 主设备：从 DDR 读激活/权重、写回结果，AXI 主口 |
| `tpu_mmio.v` | MMIO 从设备：CPU 通过寄存器配置层参数、触发计算、读状态（基址 `0x1000_5000`） |
| `soc_ddr3_top.v` / `board_top.v` | SoC 集成层与顶层（`ENABLE_TPU` / `TPU_P` / `TPU_C` 参数使能） |

当前状态：引擎 23/23 用例 bit-exact 回归通过；SoC 集成综合时序通过（93.75 MHz，WNS +0.542ns）；**未上板，待烧新位流实测**。

## 2. 目录结构

```
tpu/
├── README.md                        ← 本说明
├── doc/                             ← SoC + TPU 交接说明（10 份）
│   ├── SoC与TPU架构说明.md          ← 架构、挂载方式、数据流、复现步骤（先读这个）
│   ├── 项目交接报告_2026-10-22.md   ← 项目全貌、演进史、待办、坑位清单
│   ├── TPU卷积引擎v1~v4_设计说明    ← 引擎五代演进（历史回溯）
│   ├── soc_tpu_synth_report.md      ← SoC 集成综合报告（时序/资源）
│   ├── TPU权重文件格式规范.md       ← 权重/偏置/激活布局（已冻结）
│   ├── TPU权重接口格式规范.md       ← 寄存器接口定义
│   └── 发给权重侧_TPU权重对接说明.md← 给模型/权重侧同事的对接文档（已冻结）
├── rtl/                             ← TPU 相关 RTL（v9 引擎 + SoC 集成层）
│   ├── tpu_conv_engine.v  tpu_dma.v  tpu_mmio.v
│   └── soc_ddr3_top.v  board_top.v
├── sim/                             ← 仿真与综合复现
│   ├── tpu/                         ← 引擎级回归（23 用例）+ 各模块 TB
│   │   ├── run_regress.py  gen_stim.py  run_mmio_check.py
│   │   ├── tb_tpu_conv.v  tb_tpu_dma.v  tb_tpu_layout.v  tb_tpu_mmio.v
│   │   │   tb_tpu_perf.v  tb_tpu_wdma.v  tb_bus_tpu.v
│   │   └── synth/                   ← 引擎单独综合/PnR 评估（PDS）
│   │       ├── run_synth2.tcl  impl.tcl  run_pnr_v15.tcl
│   │       ├── tpu_synth_wrap.v  tpu_pnr_wrap_v15.v  tpu_conv_engine_pnr_snap.v
│   │       └── tpu.fdc  tpu_pnr.fdc
│   ├── tpu_soc/                     ← SoC 级协同仿真（真 CPU + DDR）
│   │   ├── run.py  tb_tpu_soc.v
│   └── soc/synth/                   ← 板级综合工程（PDS 2022.2-SP6.4）
│       ├── run_soc_tpu.tcl  run_soc_tpu32.tcl
│       ├── board_top_notpu.v  board_top_tpu32.v
│       └── soc_tpu_run/soc_tpu.pds  impl.tcl
└── tests/tpu_conv/                  ← C 软硬协同测试程序（打印 TPU_SOC_PASS）
    ├── main.c  startup.S  linker.ld  build.py
```

## 3. 验证状态

| 项 | 状态 | 说明 |
|---|---|---|
| 引擎功能回归 | ✅ 23/23 bit-exact | `sim/tpu/run_regress.py`，覆盖 3×3/1×1、stride 1/2、pad 0/1、ReLU、多通道块/组 |
| MMIO/DMA 总线检查 | ✅ 通过 | `sim/tpu/run_mmio_check.py` 等 |
| SoC 级协同仿真 | ✅ 通过 | `sim/tpu_soc/run.py`，真 CPU 从 DDR 启动，打印 `TPU_SOC_PASS`（约 7~8 分钟） |
| SoC 综合时序 | ✅ 通过 | seed_3 布线后 WNS **+0.542ns**，TNS=0，等效 Fmax ≈ 98.7 MHz，主频 93.75 MHz |
| 资源余量 | ✅ 充足 | LUT 18% / FF 8% / DRM 23% / APM(DSP) 19%（全 SoC） |
| 上板实测 | ⏳ 未做 | 待烧新位流（板上仍为旧 CNN 位流） |

## 4. 快速复现

**引擎级回归**（需 Python + numpy + iverilog；Windows 下 iverilog 依赖 MSYS2 DLL，需把 MSYS2 bin 加入 PATH）：

```bash
python -X utf8 sim/tpu/run_regress.py [--quick] [--keep]
```

**SoC 级协同仿真**：

```bash
python sim/tpu_soc/run.py
```

> ⚠️ 协同仿真需要 CPU 核、总线、UART、DDR 桥等其余 SoC 模块（`run.py` 编译清单含
> `myriscv/` 下 19 个模块 + `sim/uart_echo/tb/echo_bram.v` + `sim/br_ex_redirect/mul_behavioral.v`），
> 本仓库仅含 TPU 部分，需在完整主仓库中运行；`tests/tpu_conv/build.py` 依赖
> `xpack-riscv-none-elf-gcc-15.2.0-1` 工具链。

**引擎单独综合评估**（PDS，可选，用于查引擎频率/资源）：

```bash
# 综合（P32C16 配置示例）
pds_shell -file sim/tpu/synth/run_synth2.tcl
# 布局布线（v15 wrapper）
pds_shell -file sim/tpu/synth/run_pnr_v15.tcl
```

**综合上板**：紫光同创 PDS 2022.2-SP6.4 打开 `sim/soc/synth/soc_tpu_run/soc_tpu.pds`（top = board_top，ENABLE_TPU=1），或运行 `run_soc_tpu.tcl` 一键综合。
> ⚠️ `.pds` / `.tcl` 内为绝对路径 `D:/riscv/RISCV/...`，clone 后需改为本地路径（或保持相同目录结构）。

## 5. 版本说明

- 引擎当前版本：**v9**（`rtl/tpu_conv_engine.v` 文件头标注；即原"v5+S3"综合化改造系列的迭代结果，含 S2 乘积寄存器 + 加法树中间插级 tm）。
- 历史版本 v1~v4 见 `doc/TPU卷积引擎v1~v4_设计说明_*.md`，仅作演进回溯。
- 接口格式已冻结（权重 int8 / 偏置 int32 / 布局 O_CB_K_I / P16×C8 / cout≤256 / cin≤64），后续优化不改变接口，见 `doc/发给权重侧_TPU权重对接说明.md`。

## 6. 来源与许可

本仓库内容从开发主仓库导出，与主仓库 `doc/tpu/`、`myriscv/`、`sim/tpu/`、`sim/tpu_soc/`、`sim/soc/synth/`、`tests/tpu_conv/` 对应；如需 CPU 核、CNN 引擎、摄像头等其余 SoC 代码，请索取完整主仓库。
