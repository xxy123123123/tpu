"""TPU 卷积引擎回归脚本：生成激励 -> 编译 -> 仿真 -> 汇总。

用法:
    python sim/tpu/run_regress.py                 # 跑全部用例
    python sim/tpu/run_regress.py --quick         # 只跑快速子集
    python sim/tpu/run_regress.py --keep          # 保留中间产物

用例覆盖：P/C 规模、stride 1/2、pad 0/1、relu 0/1、方形/窄图/非方图、
ncb(输入通道块)=1/2、npg(输出通道组)=1/2。
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # D:/riscv/RISCV
SIM = ROOT / 'sim' / 'tpu'
NCBMAX = 8      # 行缓冲的通道块维度上限（需 >= 所有用例的 CIN/C）

CASES = [
    # name,        P,  C,  H,  W, CIN, COUT, stride, pad, shift, relu, ksize
    ('min_p1c1',   1,  1,  4,  4,   1,   1,   1, 1,  6, 1, 3),
    ('p1c1_pad0',  1,  1,  6,  6,   1,   1,   1, 0, 10, 1, 3),
    ('p1c1_w5',    1,  1,  6,  5,   1,   1,   1, 1, 10, 1, 3),
    ('p4c4_base',  4,  4,  8,  8,   8,   8,   1, 1,  6, 1, 3),
    ('p4c4_s2',    4,  4,  8,  8,   8,   8,   2, 1,  6, 1, 3),
    ('p4c4_pad0',  4,  4,  8,  8,   8,   8,   1, 0,  6, 1, 3),
    ('p4c4_nrelu', 4,  4,  8,  8,   8,   8,   1, 1,  6, 0, 3),
    ('p4c4_ncb1',  4,  4,  8,  8,   4,   4,   1, 1,  6, 1, 3),
    ('p4c4_tall',  4,  4, 12,  5,   8,   4,   1, 1,  7, 1, 3),
    ('p1c4_ncb2',  1,  4,  6,  6,   8,   1,   1, 1, 10, 1, 3),
    ('p4c1_npg2',  4,  1,  6,  6,   1,   8,   1, 1, 10, 1, 3),
    ('p8c8',       8,  8, 16, 16,  16,  16,   1, 1,  8, 1, 3),
    ('p8c8_s2',    8,  8, 16, 16,  16,  16,   2, 1,  8, 1, 3),
    ('p16c8',     16,  8, 32, 32,  16,  32,   1, 1, 10, 1, 3),
    # v2 新增：多输入通道块（行缓冲块维度）、多输出通道组（权重乒乓）
    ('p8c8_ncb4',  8,  8, 12, 12,  32,  16,   1, 1,  8, 1, 3),
    ('p8c8_ncb4s2', 8, 8, 12, 12,  32,  16,   2, 1,  8, 1, 3),
    ('p4c4_npg4',  4,  4,  8,  8,   4,  16,   1, 1,  6, 1, 3),
    ('p4c4_npg4s2', 4, 4, 10, 10,   8,  16,   2, 1,  6, 1, 3),
    # v3 新增：1x1 卷积（含 stride/pad 组合），及 npg=8 验证跨 pg 行缓冲保留
    ('p4c4_k1',    4,  4,  8,  8,   8,   8,   1, 0,  6, 1, 1),
    ('p4c4_k1p1',  4,  4,  8,  8,   8,   8,   1, 1,  6, 1, 1),
    ('p8c8_k1s2',  8,  8, 16, 16,  16,  16,   2, 1,  8, 1, 1),
    ('p8c8_k1ncb4', 8, 8, 12, 12,  32,  16,   1, 0,  8, 1, 1),
    ('p4c4_npg8',  4,  4,  8,  8,   4,  32,   1, 1,  6, 1, 3),
]

QUICK = {'min_p1c1', 'p4c4_base', 'p8c8'}


def sh(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, shell=True,
                          capture_output=True, text=True, encoding='utf-8',
                          errors='replace')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--rtl', default=None,
                    help='被测 RTL 路径（默认 myriscv/tpu_conv_engine.v）；'
                         '用于对比 v6 等候选版本')
    a = ap.parse_args()

    py = sys.executable
    rtl = Path(a.rtl).resolve() if a.rtl else (ROOT / 'myriscv' / 'tpu_conv_engine.v')
    work = SIM / 'regress'
    work.mkdir(parents=True, exist_ok=True)

    passed, failed = [], []
    for (name, P, C, H, W, CIN, COUT, stride, pad, shift, relu, ksize) in CASES:
        if a.quick and name not in QUICK:
            continue
        d = work / name
        # 不做整目录删除（沙箱会拦截批量删除），直接覆盖同名文件即可
        d.mkdir(parents=True, exist_ok=True)

        gen = sh(f'"{py}" -X utf8 "{SIM}/gen_stim.py" --out "{d}" '
                 f'--P {P} --C {C} --H {H} --W {W} --CIN {CIN} --COUT {COUT} '
                 f'--stride {stride} --pad {pad} --ksize {ksize} '
                 f'--shift {shift} --relu {relu} '
                 f'--ncbmax {NCBMAX}', cwd=ROOT)
        if gen.returncode != 0:
            failed.append((name, 'GEN', gen.stderr[-200:]))
            continue

        cc = sh(f'iverilog -g2012 -I "{d}" -o "{d}/tb.vvp" '
                f'"{SIM}/tb_tpu_conv.v" "{rtl}"', cwd=ROOT)
        if cc.returncode != 0:
            failed.append((name, 'COMPILE', cc.stderr[-300:]))
            continue

        rr = sh(f'vvp "{d}/tb.vvp"', cwd=d)
        out = rr.stdout + rr.stderr
        line = next((l for l in out.splitlines() if 'RESULT' in l), '(no RESULT)')
        perf = next((l for l in out.splitlines() if 'PERF' in l), '')
        ok = line.startswith('RESULT: PASS')
        (passed if ok else failed).append((name, line.strip(), perf.strip()))
        print(f'[{"PASS" if ok else "FAIL"}] {name:12s} {line.strip()}')
        if perf:
            print(f'         {perf.strip()}')

    print()
    print(f'通过 {len(passed)} / {len(passed) + len(failed)}')
    if failed:
        print('失败项:')
        for f in failed:
            print('  ', f)
        return 1
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
