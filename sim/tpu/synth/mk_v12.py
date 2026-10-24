"""v12 = v11 + 给 s1b_act / s1b_w 补复位值。

v11 失败根因（综合日志实锤）：
    W: Sdm-2014: Feedback mux created for signal 's1b_w'.
       It is possible a async set/reset assignment for this signal is missing.
4096 bit 的 s1b_w 没有复位值 -> 综合器为每个 bit 造反馈 MUX ->
乘法操作数不再是「干净寄存器」-> PDS 的 APM 4x8x8 SIMD 打包失效，
APM 从 128 暴涨到 512，LUT 8884->14690，Fmax 反而从 148.77 掉到 143.02。

修复：在 S1b 级显式给出复位值，让 s1b_act / s1b_w 成为干净的寄存器。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'myriscv' / 'tpu_conv_engine_v11.v'
DST = ROOT / 'myriscv' / 'tpu_conv_engine_v12.v'

OLD = """        if (!resetn) begin
            s1b_valid <= 1'b0; s1b_first <= 1'b0; s1b_last <= 1'b0;
            s1b_pgfin <= 1'b0; s1b_fin <= 1'b0; s1b_chbase <= 16'd0;
        end else if (!gfreeze) begin"""

NEW = """        if (!resetn) begin
            s1b_valid <= 1'b0; s1b_first <= 1'b0; s1b_last <= 1'b0;
            s1b_pgfin <= 1'b0; s1b_fin <= 1'b0; s1b_chbase <= 16'd0;
            // v12 关键：显式复位值。v11 缺失时综合器给 4096bit 的 s1b_w 造了
            // 反馈 MUX（Sdm-2014 警告），操作数不再是干净寄存器，
            // APM 的 4x8x8 SIMD 打包随即失效（APM 128 -> 512）。
            s1b_act   <= {C*8{1'b0}};
            s1b_w     <= {P*C*8{1'b0}};
        end else if (!gfreeze) begin"""


def main():
    txt = SRC.read_text(encoding='utf-8')
    if OLD not in txt:
        raise SystemExit('[mk_v12] 未匹配到 S1b 复位块')
    txt = txt.replace(OLD, NEW, 1)
    DST.write_text(txt, encoding='utf-8')
    print(f'[mk_v12] wrote {DST}  ({len(txt.splitlines())} lines)')


if __name__ == '__main__':
    main()
