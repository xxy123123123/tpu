"""汇总各版本 TPU PnR 结果：Fmax / WNS / 关键路径 / 逻辑-布线占比。

用法：
    py -3 sim/tpu/synth/summarize_pnr.py                       # 打印 Markdown 表
    py -3 sim/tpu/synth/summarize_pnr.py --out doc/tpu/pnr对比.md

扫描 sim/tpu/synth/pnr_run*/report_timing/*.rtr（含 pnr_run 基线目录）。
"""
import argparse
import re
from pathlib import Path

SYNTH = Path(__file__).resolve().parent

RE_FMAX = re.compile(
    r'^\s*clk\s+\d+\.\d+\s*MHz\s+([\d.]+)\s*MHz\s+[\d.]+\s+([\d.]+)\s+(-?[\d.]+)',
    re.M)
RE_START = re.compile(r'^Startpoint\s*:\s*(\S+)', re.M)
RE_END = re.compile(r'^Endpoint\s*:\s*(\S+)', re.M)
RE_ARRIVAL = re.compile(
    r'Data arrival time\s+([\d.]+)\s+Logic Levels:\s*(\d+)', re.M)
RE_SPLIT = re.compile(
    r'Logic:\s*([\d.]+)ns\(([\d.]+)%\),\s*Route:\s*([\d.]+)ns', re.M)


def parse(rtr: Path):
    t = rtr.read_text(encoding='utf-8', errors='replace')
    m = RE_FMAX.search(t)
    fmax = float(m.group(1)) if m else None
    wns = float(m.group(3)) if m else None
    sp = RE_START.findall(t)
    ep = RE_END.findall(t)
    arr = RE_ARRIVAL.search(t)
    spl = RE_SPLIT.search(t)
    return {
        'fmax': fmax,
        'wns': wns,
        'start': sp[0] if sp else '-',
        'end': ep[0] if ep else '-',
        'arrival': float(arr.group(1)) if arr else None,
        'levels': int(arr.group(2)) if arr else None,
        'logic_ns': float(spl.group(1)) if spl else None,
        'route_ns': float(spl.group(3)) if spl else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    rows = []
    for d in sorted(SYNTH.glob('pnr_run*')):
        rtrs = sorted((d / 'report_timing').glob('*.rtr'))
        if not rtrs:
            continue
        name = d.name.replace('pnr_run', '').lstrip('_') or 'v5+S3(基线)'
        r = parse(rtrs[0])
        rows.append((name, r))

    lines = ['| 版本 | Fmax (MHz) | WNS (ns) | 关键路径 | Levels | 到达时间 (ns) | Logic/Route (ns) |',
             '|---|---:|---:|---|---:|---:|---:|']
    for name, r in rows:
        path = f"{r['start']} → {r['end']}"
        lr = (f"{r['logic_ns']:.2f}/{r['route_ns']:.2f}"
              if r['logic_ns'] is not None else '-')
        lines.append(
            f"| {name} | {r['fmax']:.2f} | {r['wns']:.2f} | {path} | "
            f"{r['levels']} | {r['arrival']:.2f} | {lr} |")

    out = '\n'.join(lines)
    print(out)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(out + '\n', encoding='utf-8')
        print(f'\n已写入 {p}')


if __name__ == '__main__':
    main()
