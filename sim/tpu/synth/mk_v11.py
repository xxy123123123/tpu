"""在 v10 基础上生成 v11：在 S1 与 S2a 之间插一级（S1b），把乘法操作数打一拍。

依据：pnr_run_v10 布局后关键路径
    Startpoint u/lmem_7/gopdrm_36k/CLKB[0] -> Endpoint u/N1227/gopapm/X[6]
    Data arrival 8.597ns，Logic Levels 1，Logic 1.867(38.8%) / Route 2.946(61.2%)
路径明细：
    BRAM tco 1.644 -> net 0.692 -> LUT(s1_zero MUX) 0.223 -> net act_eff[14] 2.254 -> APM X
只剩 1 级逻辑，但 61% 是布线：act_eff 扇出只有 7，说明不是扇出太大，
而是 **lmem BRAM 与 APM 的物理距离太远**，且 BRAM 位置固定、无法挪。

对策：把两个乘法操作数（激活 act_eff、权重 s1_w）再寄存一拍。
    普通 FF 不受 BRAM 位置约束，PDS 可以把它摆到 APM 附近，
    于是「BRAM -> APM」这一条长 net 被切成两段短 net。
代价：P*C*8 + C*8 = 4224 个 FF，以及每像素多 1 拍延迟（吞吐仍净赚，频率涨得更多）。

顺带把 s1_zero 的 MUX 吸收进 s1_act 寄存器：
    s1_act 与 lmem 读出本来就同拍对齐，改用「下一拍的 s1_zero 值」在寄存时就选通，
    于是 act_eff 变成纯寄存器输出，省掉路径上那 0.223ns 的 LUT。
    s1_zero_nxt = issue ? !win_ok : s1_zero    —— 正是 s1_zero 在同拍赋值后的新值

用法：
    py -3 sim/tpu/synth/mk_v11.py                # 生成 myriscv/tpu_conv_engine_v11.v
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # .../sim/tpu/synth -> RISCV
SRC = ROOT / 'myriscv' / 'tpu_conv_engine_v10.v'
DST = ROOT / 'myriscv' / 'tpu_conv_engine_v11.v'

# ---- 1. 声明区：新增 S1b 级寄存器 ----
OLD_DECL = """    // S2a（乘积拍）控制伴随信号：与 pr 同步延迟一拍
    reg              s2a_valid, s2a_first, s2a_last, s2a_pgfin, s2a_fin;
    reg [15:0]       s2a_chbase;"""

NEW_DECL = """    // S1b（乘法操作数拍，v11 新增）：把 act_eff / s1_w 再寄存一拍，
    // 让 PDS 能把它们摆到 APM 附近，切断 BRAM -> APM 的长 net
    reg              s1b_valid, s1b_first, s1b_last, s1b_pgfin, s1b_fin;
    reg [15:0]       s1b_chbase;
    reg [C*8-1:0]    s1b_act;                // 激活（已含 zero 选通）
    reg [P*C*8-1:0]  s1b_w;                  // 权重（扁平向量，不会被推断成 RAM）

    // S2a（乘积拍）控制伴随信号：与 pr 同步延迟一拍
    reg              s2a_valid, s2a_first, s2a_last, s2a_pgfin, s2a_fin;
    reg [15:0]       s2a_chbase;"""

# ---- 2. act_eff：MUX 吸收进 s1_act 后，这里变成纯寄存器输出 ----
OLD_ACTEFF = """    // 乘法阵列（S1 寄存器 -> 组合乘法 -> S2 拍累加）
    wire [C*8-1:0] act_eff = s1_zero ? {C*8{1'b0}} : s1_act;"""

NEW_ACTEFF = """    // 乘法阵列（S1b 寄存器 -> 组合乘法 -> S2 拍累加）
    // v11：s1_zero 的选通已前移到 s1_act 的寄存时刻，故 act_eff 是纯寄存器输出
    wire [C*8-1:0] act_eff = s1b_act;
    // s1_act 与 lmem 读出同拍对齐，用「同拍赋值后 s1_zero 的新值」在寄存时选通
    wire s1_zero_nxt = issue ? !win_ok : s1_zero;"""

# ---- 3. 乘法输入改用 S1b 级 ----
OLD_PROD = """                assign prod[mp*C+mc] = $signed(act_eff[mc*8 +: 8])
                                     * $signed(s1_w[(mp*C+mc)*8 +: 8]);"""

NEW_PROD = """                assign prod[mp*C+mc] = $signed(act_eff[mc*8 +: 8])
                                     * $signed(s1b_w[(mp*C+mc)*8 +: 8]);"""

# ---- 4. s1_act 寄存时就把 zero 选通做掉 ----
OLD_S1ACT = "            s1_act <= lmem[lbuf_raddr];"

NEW_S1ACT = "            s1_act <= s1_zero_nxt ? {C*8{1'b0}} : lmem[lbuf_raddr];"

# ---- 5. S2a 伴随信号改从 S1b 取（整体顺延一拍）----
OLD_S2A = """            s2a_valid  <= s1_valid;
            s2a_first  <= s1_first;
            s2a_last   <= s1_last;
            s2a_pgfin  <= s1_pgfin;
            s2a_fin    <= s1_fin;
            s2a_chbase <= s1_chbase;"""

NEW_S2A = """            s2a_valid  <= s1b_valid;
            s2a_first  <= s1b_first;
            s2a_last   <= s1b_last;
            s2a_pgfin  <= s1b_pgfin;
            s2a_fin    <= s1b_fin;
            s2a_chbase <= s1b_chbase;"""

# ---- 6. 新增 S1b 流水级（插在 S1 块之后）----
OLD_ANCHOR = """    // ---------------- S2: 乘加（组合乘法 + 加法树 -> pacc / s2_acc）----------------"""

NEW_S1B = """    // ---------------- S1b: 乘法操作数打拍级（v11 新增）----------------
    // v10 布局后关键路径只剩 1 级逻辑，却仍有 8.597ns——61% 是布线：
    //   lmem BRAM(tco 1.644) -> net 0.692 -> LUT 0.223 -> net act_eff(2.254) -> APM X
    // act_eff 扇出只有 7，可见不是扇出问题，而是 BRAM 与 APM 隔得太远（BRAM 位置固定）。
    // 这里把激活与权重各再寄存一拍：普通 FF 不受 BRAM 位置约束，可被摆到 APM 旁边，
    // 一条长 net 由此拆成两段短 net。代价 4224 个 FF + 每像素 1 拍延迟。
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            s1b_valid <= 1'b0; s1b_first <= 1'b0; s1b_last <= 1'b0;
            s1b_pgfin <= 1'b0; s1b_fin <= 1'b0; s1b_chbase <= 16'd0;
        end else if (!gfreeze) begin
            s1b_valid  <= s1_valid;
            s1b_first  <= s1_first;
            s1b_last   <= s1_last;
            s1b_pgfin  <= s1_pgfin;
            s1b_fin    <= s1_fin;
            s1b_chbase <= s1_chbase;
            s1b_act    <= s1_act;
            s1b_w      <= s1_w;
        end
    end

    // ---------------- S2: 乘加（组合乘法 + 加法树 -> pacc / s2_acc）----------------"""

REPLACEMENTS = [
    ('声明区', OLD_DECL, NEW_DECL),
    ('act_eff', OLD_ACTEFF, NEW_ACTEFF),
    ('乘法输入', OLD_PROD, NEW_PROD),
    ('s1_act 选通', OLD_S1ACT, NEW_S1ACT),
    ('S2a 伴随', OLD_S2A, NEW_S2A),
    ('S1b 级', OLD_ANCHOR, NEW_S1B),
]


def main():
    txt = SRC.read_text(encoding='utf-8')
    for name, old, new in REPLACEMENTS:
        if old not in txt:
            raise SystemExit(f'[mk_v11] 未匹配到：{name}')
        txt = txt.replace(old, new, 1)
    DST.write_text(txt, encoding='utf-8')
    print(f'[mk_v11] wrote {DST}  ({len(txt.splitlines())} lines)')


if __name__ == '__main__':
    main()
