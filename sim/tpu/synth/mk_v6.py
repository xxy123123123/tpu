"""由 tpu_conv_engine.v (v5+S3) 生成 v6 候选：S2 的 C 项串行累加链 -> 显式加法树。

背景：原 S2 里 `acc_sum = acc_sum + prod[i]` 用阻塞赋值展开，语义上是 C 级串行
加法链（C=16 即 16 级 32bit 加法）。综合器未必重构成树，这是 P32C16 只有 ~76MHz
的头号嫌疑。本脚本把归约改成显式的 log2(C) 级两两归约树：
    面积不变（都是 C-1 个加法器），关键路径深度 C -> log2(C)+1。

用法：
    py -3 sim/tpu/synth/mk_v6.py                       # 生成 myriscv/tpu_conv_engine_v6.v
    py -3 sim/tpu/synth/mk_v6.py --out <path>          # 指定输出

生成的 v6 必须通过 sim/tpu 回归（功能等价）后再做综合对比。
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # .../sim/tpu/synth -> RISCV
SRC = ROOT / 'myriscv' / 'tpu_conv_engine.v'

# ---- S2 中原串行累加（要被替换掉）----
OLD_ACC = """                for (pi = 0; pi < P; pi = pi + 1) begin
                    // 首项以 bias 快照起步（pacc 无需复位，不会读回 X）
                    acc_sum = s1_first ? $signed(bias_p[pi*32 +: 32]) : pacc[pi];
                    for (ci = 0; ci < C; ci = ci + 1)
                        acc_sum = acc_sum + {{16{prod[pi*C+ci][15]}}, prod[pi*C+ci]};
                    pacc[pi] <= acc_sum;"""

NEW_ACC = """                for (pi = 0; pi < P; pi = pi + 1) begin
                    // 首项以 bias 快照起步（pacc 无需复位，不会读回 X）
                    // C 项乘积由显式加法树归约（见 g_tree）：深度 log2(C)+1 而非 C
                    acc_sum = (s1_first ? $signed(bias_p[pi*32 +: 32]) : pacc[pi])
                            + tree_sum[pi];
                    pacc[pi] <= acc_sum;"""

# ---- 加法树：插在乘法阵列 generate 之后、requant function 之前 ----
ANCHOR = "    // requant（RHA 舍入"

TREE = """    // ---------------- S2: C 项乘积归约（显式加法树）----------------
    // 原写法用阻塞赋值串行累加，语义上是 C 级加法链（C=16 -> 16 级 32bit 加法），
    // 是 v5+S3 综合只有 ~76MHz 的头号嫌疑。这里显式展开 log2(C) 级两两归约：
    //   加法器总数不变（每 lane C-1 个），关键路径深度 C -> log2(C)+1。
    // 存储形态：模块级一维 wire 数组 + generate 组合赋值（不涉及时钟，不推断 RAM）。
    localparam NL = (C <= 1) ? 0 : $clog2(C);   // 归约层数：C=16 -> 4
    wire signed [31:0] tw [0:(NL+1)*P*C-1];     // 扁平：层 l 的 lane p 元素 i -> (l*P+p)*C+i
    wire signed [31:0] tree_sum [0:P-1];        // 每 lane 的树根

    genvar g_tp, g_ti, g_lv;
    generate
        // 层 0：C 个乘积符号扩展到 32 位
        for (g_tp = 0; g_tp < P; g_tp = g_tp + 1) begin : g_t0p
            for (g_ti = 0; g_ti < C; g_ti = g_ti + 1) begin : g_t0i
                assign tw[(0*P + g_tp)*C + g_ti] =
                    {{16{prod[g_tp*C+g_ti][15]}}, prod[g_tp*C+g_ti]};
            end
        end
        // 层 1..NL：两两归约，每层元素数减半
        for (g_lv = 1; g_lv <= NL; g_lv = g_lv + 1) begin : g_tlp
            for (g_tp = 0; g_tp < P; g_tp = g_tp + 1) begin : g_tlpp
                for (g_ti = 0; g_ti < (C >> g_lv); g_ti = g_ti + 1) begin : g_tlpi
                    assign tw[(g_lv*P + g_tp)*C + g_ti] =
                        tw[((g_lv-1)*P + g_tp)*C + 2*g_ti] +
                        tw[((g_lv-1)*P + g_tp)*C + 2*g_ti + 1];
                end
            end
        end
        // 树根（C=1 时 NL=0，直接取层 0）
        for (g_tp = 0; g_tp < P; g_tp = g_tp + 1) begin : g_troot
            assign tree_sum[g_tp] = tw[(NL*P + g_tp)*C];
        end
    endgenerate

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'myriscv' / 'tpu_conv_engine_v6.v'))
    a = ap.parse_args()

    txt = SRC.read_text(encoding='utf-8')
    if OLD_ACC not in txt:
        raise SystemExit('未找到 S2 串行累加片段，源文件结构可能已变')
    if ANCHOR not in txt:
        raise SystemExit('未找到 requant 锚点，源文件结构可能已变')

    txt = txt.replace(OLD_ACC, NEW_ACC)
    txt = txt.replace(ANCHOR, TREE + ANCHOR, 1)
    # 头部版本说明
    txt = txt.replace('// TPU 卷积引擎 v5  ——', '// TPU 卷积引擎 v6  ——', 1)

    out = Path(a.out)
    out.write_text(txt, encoding='utf-8')
    print(f'已生成 {out}')


if __name__ == '__main__':
    main()
