"""在 v6（加法树）基础上生成 v7：乘法输出打一拍，S2 拆成 S2a/S2b 两级流水。

依据：pnr_run/report_timing 的布局后关键路径（P32C16, 200MHz 约束）
    Startpoint u/lmem_2/gopdrm_36k/CLKB[0]  ->  Endpoint u/s2_acc[701]/opit_0/D
    Data arrival 15.639ns（Logic 5.882 / Route 5.982，布线占 50%），Logic Levels 13
    路径骨架：BRAM(tco 1.644) -> act_eff(net 2.717, 扇出大) -> APM 乘法(1.971)
              -> 加法链/树 -> pacc -> s2_acc
即「BRAM 输出 + 乘法 + 加法树」全在一条组合路径上。v7 把 APM 乘积打一拍
（让综合器用上 APM 的输出寄存器），把乘法延迟与加法树延迟分到两级流水：
    成本：P*C*16 bit 寄存器（P32C16 = 8192 FF）+ 每像素 1 拍流水延迟
    收益：关键路径从 ~15.6ns 拆成「BRAM->乘法->pr」与「pr->加法树->pacc」两段

用法：
    py -3 sim/tpu/synth/mk_v7.py                 # 生成 myriscv/tpu_conv_engine_v7.v
生成的 v7 必须先过 sim/tpu 回归（--rtl 指定）再做 PnR 对比。
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # .../sim/tpu/synth -> RISCV
SRC = ROOT / 'myriscv' / 'tpu_conv_engine_v6.v'

# ---- [1] 乘积寄存器：插在乘法阵列之后、加法树之前 ----
ANCHOR_TREE = "    // ---------------- S2: C 项乘积归约（显式加法树）----------------"

PROD_REG = """    // ---------------- S2a: 乘积寄存器（切断 乘法 -> 加法树 的组合路径）----------------
    // 布局后关键路径显示 BRAM->act_eff(长net)->APM乘法->加法树 全在一条组合链上，
    // 这里把 APM 乘积打一拍（推断成 APM 输出寄存器），乘法与归约各占一段流水。
    // 形态：generate 块内独立标量 reg（不是数组，不会被 PDS 推断成 RAM/DRM）
    wire signed [15:0] prw [0:P*C-1];        // 乘积寄存器输出，供加法树用
    genvar g_pq;
    generate
        for (g_pq = 0; g_pq < P*C; g_pq = g_pq + 1) begin : g_prod
            reg signed [15:0] pr;
            always @(posedge clk) if (!gfreeze) pr <= prod[g_pq];
            assign prw[g_pq] = pr;
        end
    endgenerate

"""

# ---- [2] 加法树层 0 改吃寄存后的乘积 ----
OLD_T0 = """                assign tw[(0*P + g_tp)*C + g_ti] =
                    {{16{prod[g_tp*C+g_ti][15]}}, prod[g_tp*C+g_ti]};"""
NEW_T0 = """                assign tw[(0*P + g_tp)*C + g_ti] =
                    {{16{prw[g_tp*C+g_ti][15]}}, prw[g_tp*C+g_ti]};"""

# ---- [3] S2a 控制伴随信号声明 ----
OLD_DECL = """    // S3（requant 级）数据
    reg              s2_valid, s2_fin, s2_pgfin;"""
NEW_DECL = """    // S2a（乘积拍）控制伴随信号：与 pr 同步延迟一拍
    reg              s2a_valid, s2a_first, s2a_last, s2a_pgfin, s2a_fin;
    reg [15:0]       s2a_chbase;

    // S3（requant 级）数据
    reg              s2_valid, s2_fin, s2_pgfin;"""

# ---- [4] S2 单级 -> S2a/S2b 两级 ----
OLD_S2 = """    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            s2_valid <= 1'b0; s2_fin <= 1'b0; s2_pgfin <= 1'b0; s2_chbase <= 16'd0;
        end else if (!gfreeze) begin
            // s2_valid 每拍重写（一拍进一拍出，S3 每拍消费一个）；
            // 背压时本级冻结，s2_valid 保持，恢复后由 S3 补消费
            s2_valid <= s1_valid && s1_last;
            if (s1_valid) begin
                if (s1_last) begin
                    s2_fin    <= s1_fin;
                    s2_pgfin  <= s1_pgfin;
                    s2_chbase <= s1_chbase;
                end
                for (pi = 0; pi < P; pi = pi + 1) begin
                    // 首项以 bias 快照起步（pacc 无需复位，不会读回 X）
                    // C 项乘积由显式加法树归约（见 g_tree）：深度 log2(C)+1 而非 C
                    acc_sum = (s1_first ? $signed(bias_p[pi*32 +: 32]) : pacc[pi])
                            + tree_sum[pi];
                    pacc[pi] <= acc_sum;
                    if (s1_last)
                        s2_acc[pi*32 +: 32] <= acc_sum;   // 最终累加值交 S3 requant
                end
            end
        end
    end"""

NEW_S2 = """    // ---- S2a: 控制伴随信号与 pr 同步延迟一拍 ----
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            s2a_valid <= 1'b0; s2a_first <= 1'b0; s2a_last <= 1'b0;
            s2a_pgfin <= 1'b0; s2a_fin <= 1'b0; s2a_chbase <= 16'd0;
        end else if (!gfreeze) begin
            s2a_valid  <= s1_valid;
            s2a_first  <= s1_first;
            s2a_last   <= s1_last;
            s2a_pgfin  <= s1_pgfin;
            s2a_fin    <= s1_fin;
            s2a_chbase <= s1_chbase;
        end
    end

    // ---- S2b: 加法树归约 + 像素累加 -> pacc / s2_acc ----
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            s2_valid <= 1'b0; s2_fin <= 1'b0; s2_pgfin <= 1'b0; s2_chbase <= 16'd0;
        end else if (!gfreeze) begin
            // s2_valid 每拍重写（一拍进一拍出，S3 每拍消费一个）；
            // 背压时本级冻结，s2_valid 保持，恢复后由 S3 补消费
            s2_valid <= s2a_valid && s2a_last;
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
            end
        end
    end"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'myriscv' / 'tpu_conv_engine_v7.v'))
    a = ap.parse_args()

    txt = SRC.read_text(encoding='utf-8')
    for tag, old in (('S2 单级块', OLD_S2), ('加法树层0', OLD_T0),
                     ('s2 声明', OLD_DECL), ('加法树锚点', ANCHOR_TREE)):
        if old not in txt:
            raise SystemExit(f'未找到 {tag}，v6 源文件结构可能已变')

    txt = txt.replace(OLD_S2, NEW_S2)
    txt = txt.replace(OLD_T0, NEW_T0)
    txt = txt.replace(OLD_DECL, NEW_DECL)
    txt = txt.replace(ANCHOR_TREE, PROD_REG + ANCHOR_TREE, 1)
    txt = txt.replace('// TPU 卷积引擎 v6  ——', '// TPU 卷积引擎 v7  ——', 1)

    out = Path(a.out)
    out.write_text(txt, encoding='utf-8')
    print(f'已生成 {out}')


if __name__ == '__main__':
    main()
