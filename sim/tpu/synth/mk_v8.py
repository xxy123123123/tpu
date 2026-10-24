"""在 v7（乘法打拍）基础上生成 v8：去掉 S0 地址计算里的 APM(DSP) 乘法。

依据：pnr_run_v7 布局后关键路径已不再是 MAC 数据通路，而是 S0 控制/地址路径
    Startpoint u/N90/gopapm/CLK  ->  Endpoint u/N193/gopapm/CE_Z   (11.261ns, Levels 5)
    骨架：APM(oy*stride, tco 2.650) -> row_base -> want_row -> want_real
          -> s1_valid/issue -> APM 的 CE_Z
`oy * stride` 只有 1/2 两种取值，却被综合成 25x18 乘法器（APM），光 tco 就 2.65ns。
v8 把 row_base / col_rd 的乘法换成「移位 or 直连」，并把 ky*3 换成常量选择。

用法：
    py -3 sim/tpu/synth/mk_v8.py                 # 生成 myriscv/tpu_conv_engine_v8.v
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # .../sim/tpu/synth -> RISCV
SRC = ROOT / 'myriscv' / 'tpu_conv_engine_v7.v'

OLD_ROW = """    wire signed [16:0] row_base = $signed({1'b0, oy}) * $signed({14'b0, stride_r})
                                - (pad_r ? 17'sd1 : 17'sd0);"""
NEW_ROW = """    // stride 只有 1/2 两种取值：用「移位 or 直连」代替乘法。
    // 原写法被综合器映射到 APM(DSP)，仅 tco 就 2.65ns，是 v7 布局后关键路径的起点。
    wire [16:0] oy_x2 = {oy, 1'b0};
    wire signed [16:0] row_base = $signed((stride_r == 2'd2) ? oy_x2 : {1'b0, oy})
                                - (pad_r ? 17'sd1 : 17'sd0);"""

OLD_COL = """    wire signed [16:0] col_rd = $signed({1'b0, ox}) * $signed({14'b0, stride_r})
                              + $signed({13'b0, kx}) - (pad_r ? 17'sd1 : 17'sd0);"""
NEW_COL = """    wire [16:0] ox_x2 = {ox, 1'b0};
    wire signed [16:0] col_rd = $signed((stride_r == 2'd2) ? ox_x2 : {1'b0, ox})
                              + $signed({13'b0, kx}) - (pad_r ? 17'sd1 : 17'sd0);"""

OLD_KX = "    wire [3:0] kx = ksize_r ? 4'd0 : (k - ky * 4'd3);"
NEW_KX = """    wire [3:0] kx = ksize_r ? 4'd0
                  : (k - ((ky == 4'd2) ? 4'd6 : (ky == 4'd1) ? 4'd3 : 4'd0));"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'myriscv' / 'tpu_conv_engine_v8.v'))
    a = ap.parse_args()

    txt = SRC.read_text(encoding='utf-8')
    for tag, old in (('row_base', OLD_ROW), ('col_rd', OLD_COL), ('kx', OLD_KX)):
        if old not in txt:
            raise SystemExit(f'未找到 {tag}，v7 源文件结构可能已变')

    txt = txt.replace(OLD_ROW, NEW_ROW)
    txt = txt.replace(OLD_COL, NEW_COL)
    txt = txt.replace(OLD_KX, NEW_KX)
    txt = txt.replace('// TPU 卷积引擎 v7  ——', '// TPU 卷积引擎 v8  ——', 1)

    out = Path(a.out)
    out.write_text(txt, encoding='utf-8')
    print(f'已生成 {out}')


if __name__ == '__main__':
    main()
