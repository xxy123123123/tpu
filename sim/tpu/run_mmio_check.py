"""TPU MMIO / DMA 接入验证：编译并运行三个测试台。

    py -3 sim/tpu/run_mmio_check.py            # 三个都跑
    py -3 sim/tpu/run_mmio_check.py --mmio     # 只跑 CPU 搬运端到端
    py -3 sim/tpu/run_mmio_check.py --dma      # 只跑 DMA 端到端
    py -3 sim/tpu/run_mmio_check.py --bus      # 只跑总线窗口译码
    py -3 sim/tpu/run_mmio_check.py --wdma     # 只跑权重 DMA 端到端
    py -3 sim/tpu/run_mmio_check.py --layout  # 只跑输出布局重塑
    py -3 sim/tpu/run_mmio_check.py --perf    # 只跑 stall 分类计数器

tb_tpu_mmio : 模拟 CPU 用 MMIO 配置 TPU / 灌权重 / 喂激活 / 读回输出并比对
tb_tpu_dma  : CPU 只做配置，激活从 DDR 读、结果写回 DDR，比对 DDR 内容
tb_tpu_wdma : 权重也走 DMA（CPU 一次权重搬运都不做），cout=32 两组乒乓
tb_tpu_layout: 输出布局重塑，验证多层可直接串联（逐字节校验落位）
tb_tpu_perf  : stall 分类计数器口径（完备性 + 欠喂/背压被正确归类）
tb_tpu_layout(TWO_LEVEL): 两级读仲裁，复现 soc_ddr3_top 的实际接法
tb_bus_tpu  : 验证 0x1000_5000 窗口命中 TPU，且不误选 CNN / RAM
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / 'sim' / 'tpu'
BUILD = SIM / 'mmio_build'

CASES = {
    'mmio': dict(
        top='tb_tpu_mmio',
        srcs=['sim/tpu/tb_tpu_mmio.v', 'myriscv/tpu_mmio.v',
              'myriscv/tpu_dma.v', 'myriscv/tpu_conv_engine.v'],
    ),
    'dma': dict(
        top='tb_tpu_dma',
        srcs=['sim/tpu/tb_tpu_dma.v', 'myriscv/tpu_mmio.v',
              'myriscv/tpu_dma.v', 'myriscv/tpu_conv_engine.v'],
    ),
    'wdma': dict(
        top='tb_tpu_wdma',
        srcs=['sim/tpu/tb_tpu_wdma.v', 'myriscv/tpu_mmio.v',
              'myriscv/tpu_dma.v', 'myriscv/tpu_conv_engine.v',
              'myriscv/preprocess_ddr_read_arbiter.v'],
    ),
    'layout': dict(
        top='tb_tpu_layout',
        srcs=['sim/tpu/tb_tpu_layout.v', 'myriscv/tpu_mmio.v',
              'myriscv/tpu_dma.v', 'myriscv/tpu_conv_engine.v',
              'myriscv/preprocess_ddr_read_arbiter.v'],
    ),
    'perf': dict(
        top='tb_tpu_perf',
        srcs=['sim/tpu/tb_tpu_perf.v', 'myriscv/tpu_conv_engine.v'],
    ),
    'wtport': dict(
        top='tb_tpu_layout',
        defs=['-DTWO_LEVEL'],   # 两级读仲裁，复现 soc_ddr3_top 的实际接法
        srcs=['sim/tpu/tb_tpu_layout.v', 'myriscv/tpu_mmio.v',
              'myriscv/tpu_dma.v', 'myriscv/tpu_conv_engine.v',
              'myriscv/preprocess_ddr_read_arbiter.v'],
    ),
    'bus': dict(
        top='tb_bus_tpu',
        srcs=['sim/tpu/tb_bus_tpu.v', 'myriscv/data_bus_interconnect.v'],
    ),
}


def main():
    ap = argparse.ArgumentParser()
    for k in CASES:
        ap.add_argument(f'--{k}', action='store_true')
    ap.add_argument('--shared', action='store_true',
                    help='权重与激活共用一条读口（对照实验，默认独立口）')
    a = ap.parse_args()
    want = [k for k in CASES if getattr(a, k)] or list(CASES)

    BUILD.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in want:
        case = CASES[name]
        top = case['top']
        exe = BUILD / f'{top}.vvp'
        cmd = ['iverilog', '-g2012', '-I', str(ROOT / 'myriscv'), '-s', top,
               '-o', str(exe)]
        if a.shared:
            cmd.append('-DWT_SHARED')
        cmd += case.get('defs', [])
        cmd += [str(ROOT / s) for s in case['srcs']]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        if r.returncode != 0:
            print(f'[{top}] 编译失败:\n{r.stderr[:1500]}')
            failed.append(name)
            continue
        r = subprocess.run(['vvp', str(exe)], cwd=ROOT, capture_output=True,
                           text=True, encoding='utf-8', timeout=600)
        out = r.stdout
        print(f'----- {top} -----')
        print(out.rstrip()[:2500])
        if 'PASS' not in out or 'FAIL (' in out:
            failed.append(name)
        print()

    print('=' * 46)
    if failed:
        print(f'FAILED: {", ".join(failed)}')
        return 1
    print('ALL PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
