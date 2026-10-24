"""在 v8 基础上生成 v9：加法树中间插一级寄存器，把归约拆成两段流水。

依据：pnr_run_v8 布局后关键路径
    Startpoint u/N2504/gopapm/CLK(pr) -> Endpoint u/pacc[26][31]/D
    Data arrival 11.396ns，Levels 11，Logic 3.746 / Route 3.783（布线仍占 50%）
即「pr -> 4 级加法树 -> pacc」整段太长。v9 在树的中间（SPLIT = NL/2）插一级
寄存器 tm，把 C=16 的 4 级归约拆成 2+2 两段流水：
    成本：P*(C>>SPLIT) 个 32bit 寄存器（P32C16 = 32*4*32 = 4096 FF）+ 每像素 1 拍
    收益：单段归约深度 4 -> 2，并切断树中间的进位链

用法：
    py -3 sim/tpu/synth/mk_v9.py                 # 生成 myriscv/tpu_conv_engine_v9.v
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # .../sim/tpu/synth -> RISCV
SRC = ROOT / 'myriscv' / 'tpu_conv_engine_v8.v'

A_TREE = "    // ---------------- S2: C 项乘积归约（显式加法树）----------------"
A_REQ = "    // requant（RHA 舍入"

NEW_TREE = """    // ---------------- S2: C 项乘积归约（显式加法树 + 中间打拍）----------------
    // v8 布局后关键路径就是 pr -> 加法树 -> pacc（11.4ns, Levels 11，布线占 50%）。
    // v9 在树的中间插一级寄存器 tm，把归约拆成「前半 + 后半」两段流水。
    localparam NL    = (C <= 1) ? 0 : $clog2(C);   // 归约层数：C=16 -> 4
    localparam SPLIT = NL / 2;                     // 前半层数（中间寄存器插在这里）
    wire signed [31:0] tw [0:(NL+1)*P*C-1];        // 层 l 的 lane p 元素 i -> (l*P+p)*C+i
    wire signed [31:0] twm [0:P*(C>>SPLIT)-1];     // 前半结果寄存后的输出
    wire signed [31:0] tree_sum [0:P-1];           // 每 lane 的树根

    genvar g_tp, g_ti, g_lv, g_ml, g_mn;
    generate
        // 层 0：C 个乘积符号扩展到 32 位（吃寄存后的乘积 prw）
        for (g_tp = 0; g_tp < P; g_tp = g_tp + 1) begin : g_t0p
            for (g_ti = 0; g_ti < C; g_ti = g_ti + 1) begin : g_t0i
                assign tw[(0*P + g_tp)*C + g_ti] =
                    {{16{prw[g_tp*C+g_ti][15]}}, prw[g_tp*C+g_ti]};
            end
        end
        // 前半层 1..SPLIT
        for (g_lv = 1; g_lv <= SPLIT; g_lv = g_lv + 1) begin : g_tlp
            for (g_tp = 0; g_tp < P; g_tp = g_tp + 1) begin : g_tlpp
                for (g_ti = 0; g_ti < (C >> g_lv); g_ti = g_ti + 1) begin : g_tlpi
                    assign tw[(g_lv*P + g_tp)*C + g_ti] =
                        tw[((g_lv-1)*P + g_tp)*C + 2*g_ti] +
                        tw[((g_lv-1)*P + g_tp)*C + 2*g_ti + 1];
                end
            end
        end
        // 中间寄存器：前半结果打一拍（generate 块内标量 reg，不会被推断成 RAM）
        for (g_ml = 0; g_ml < P; g_ml = g_ml + 1) begin : g_midp
            for (g_mn = 0; g_mn < (C >> SPLIT); g_mn = g_mn + 1) begin : g_midi
                reg signed [31:0] tm;
                always @(posedge clk) if (!gfreeze) tm <= tw[(SPLIT*P + g_ml)*C + g_mn];
                assign twm[g_ml*(C>>SPLIT) + g_mn] = tm;
            end
        end
        // 后半层 SPLIT+1..NL
        for (g_lv = SPLIT + 1; g_lv <= NL; g_lv = g_lv + 1) begin : g_tlq
            for (g_tp = 0; g_tp < P; g_tp = g_tp + 1) begin : g_tlqp
                for (g_ti = 0; g_ti < (C >> g_lv); g_ti = g_ti + 1) begin : g_tlqi
                    if (g_lv == SPLIT + 1) begin
                        assign tw[(g_lv*P + g_tp)*C + g_ti] =
                            twm[g_tp*(C>>SPLIT) + 2*g_ti] +
                            twm[g_tp*(C>>SPLIT) + 2*g_ti + 1];
                    end else begin
                        assign tw[(g_lv*P + g_tp)*C + g_ti] =
                            tw[((g_lv-1)*P + g_tp)*C + 2*g_ti] +
                            tw[((g_lv-1)*P + g_tp)*C + 2*g_ti + 1];
                    end
                end
            end
        end
        // 树根（无后半层时直接取中间寄存器）
        for (g_tp = 0; g_tp < P; g_tp = g_tp + 1) begin : g_troot
            if (NL > SPLIT) begin
                assign tree_sum[g_tp] = tw[(NL*P + g_tp)*C];
            end else begin
                assign tree_sum[g_tp] = twm[g_tp*(C>>SPLIT)];
            end
        end
    endgenerate

"""

# ---- 伴随信号声明 ----
OLD_DECL = """    // S2a（乘积拍）控制伴随信号：与 pr 同步延迟一拍
    reg              s2a_valid, s2a_first, s2a_last, s2a_pgfin, s2a_fin;
    reg [15:0]       s2a_chbase;"""
NEW_DECL = """    // S2a（乘积拍）控制伴随信号：与 pr 同步延迟一拍
    reg              s2a_valid, s2a_first, s2a_last, s2a_pgfin, s2a_fin;
    reg [15:0]       s2a_chbase;
    // S2b（加法树前半拍）控制伴随信号：与 tm 同步再延迟一拍
    reg              s2b_valid, s2b_first, s2b_last, s2b_pgfin, s2b_fin;
    reg [15:0]       s2b_chbase;"""

# ---- S2a 之后插入 S2b 伴随级 ----
A_S2B_ANCHOR = """    // ---- S2b: 加法树归约 + 像素累加 -> pacc / s2_acc ----"""

S2B_STAGE = """    // ---- S2b: 加法树前半伴随信号（与 tm 同步延迟一拍）----
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            s2b_valid <= 1'b0; s2b_first <= 1'b0; s2b_last <= 1'b0;
            s2b_pgfin <= 1'b0; s2b_fin <= 1'b0; s2b_chbase <= 16'd0;
        end else if (!gfreeze) begin
            s2b_valid  <= s2a_valid;
            s2b_first  <= s2a_first;
            s2b_last   <= s2a_last;
            s2b_pgfin  <= s2a_pgfin;
            s2b_fin    <= s2a_fin;
            s2b_chbase <= s2a_chbase;
        end
    end

"""

# ---- 累加级：s2a_* -> s2b_*（树根现在晚一拍到齐）----
OLD_ACC = """            s2_valid <= s2a_valid && s2a_last;
            if (s2a_valid) begin
                if (s2a_last) begin
                    s2_fin    <= s2a_fin;
                    s2_pgfin  <= s2a_pgfin;
                    s2_chbase <= s2a_chbase;
                end
                for (pi = 0; pi < P; pi = pi + 1) begin
                    // 首项以 bias 快照起步（pacc 无需复位，不会读回 X）
                    // C 项乘积由显式加法树归约（见 g_t0p/g_tlp）：深度 log2(C)+1
                    acc_sum = (s2a_first ? $signed(bias_p[pi*32 +: 32]) : pacc[pi])
                            + tree_sum[pi];
                    pacc[pi] <= acc_sum;
                    if (s2a_last)
                        s2_acc[pi*32 +: 32] <= acc_sum;   // 最终累加值交 S3 requant
                end
            end"""
NEW_ACC = """            s2_valid <= s2b_valid && s2b_last;
            if (s2b_valid) begin
                if (s2b_last) begin
                    s2_fin    <= s2b_fin;
                    s2_pgfin  <= s2b_pgfin;
                    s2_chbase <= s2b_chbase;
                end
                for (pi = 0; pi < P; pi = pi + 1) begin
                    // 首项以 bias 快照起步（pacc 无需复位，不会读回 X）
                    // C 项乘积由「前半+后半」两段加法树归约（见 g_tlp/g_tlq）
                    acc_sum = (s2b_first ? $signed(bias_p[pi*32 +: 32]) : pacc[pi])
                            + tree_sum[pi];
                    pacc[pi] <= acc_sum;
                    if (s2b_last)
                        s2_acc[pi*32 +: 32] <= acc_sum;   // 最终累加值交 S3 requant
                end
            end"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'myriscv' / 'tpu_conv_engine_v9.v'))
    a = ap.parse_args()

    txt = SRC.read_text(encoding='utf-8')
    for tag, old in (('加法树锚点', A_TREE), ('requant 锚点', A_REQ),
                     ('s2a 声明', OLD_DECL), ('S2b 锚点', A_S2B_ANCHOR),
                     ('累加块', OLD_ACC)):
        if old not in txt:
            raise SystemExit(f'未找到 {tag}，v8 源文件结构可能已变')

    # 整段替换加法树（从锚点到 requant 之前）
    i = txt.index(A_TREE)
    j = txt.index(A_REQ)
    txt = txt[:i] + NEW_TREE + txt[j:]

    txt = txt.replace(OLD_DECL, NEW_DECL)
    txt = txt.replace(A_S2B_ANCHOR, S2B_STAGE + A_S2B_ANCHOR, 1)
    txt = txt.replace(OLD_ACC, NEW_ACC)
    txt = txt.replace('// TPU 卷积引擎 v8  ——', '// TPU 卷积引擎 v9  ——', 1)

    out = Path(a.out)
    out.write_text(txt, encoding='utf-8')
    print(f'已生成 {out}')


if __name__ == '__main__':
    main()
