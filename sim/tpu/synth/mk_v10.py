"""在 v9 基础上生成 v10：去掉权重地址里的 APM 乘法（cb*nk -> 移位加）。

依据：pnr_run_v9 布局后关键路径
    Startpoint u/ksize_r/opit_0/CLK -> Endpoint u/g_wmem[28].wbank/gopdrm_18k/ADB[8]
    Data arrival 10.737ns，Logic Levels 2，Logic 3.128 / Route 3.851
路径中段：
    ksize_r -> ksize_r_inv -> klast[3] -> APM_311_306/P_0[3] (td 2.754)
                                        -> net(fanout=32) 3.139 -> wmem ADB1[8]
即只剩两级逻辑，其中 2.754ns 是一个 APM（DSP）乘法、3.139ns 是扇出 32 的广播走线。
追到 RTL 是权重地址：
    wire [KW-1:0] widx = (cb * nk) + {12'b0, k};      // nk = ksize_r ? 1 : 9
nk 只有 1 / 9 两种取值，却占用了一个 APM。改为常数移位-加乘法：
    nk == 1 -> widx = cb + k
    nk == 9 -> widx = (cb << 3) + cb + k
纯 LUT 进位链实现，不消耗 APM，且逻辑级数可控。

用法：
    py -3 sim/tpu/synth/mk_v10.py                # 生成 myriscv/tpu_conv_engine_v10.v
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # .../sim/tpu/synth -> RISCV
SRC = ROOT / 'myriscv' / 'tpu_conv_engine_v9.v'
DST = ROOT / 'myriscv' / 'tpu_conv_engine_v10.v'

OLD = "    wire [KW-1:0]      widx = (cb * nk) + {12'b0, k};"

NEW = """    // ---------------- v10: 权重地址去掉 APM 乘法 ----------------
    // v9 布局后关键路径就是它：cb*nk 被映射到 APM（td 2.754ns），输出再以
    // fanout=32 广播到所有 wmem bank（net 3.139ns），两级逻辑吃掉 10.7ns。
    // nk 只有 1 / 9 两种取值，用常数移位-加乘法替代，完全等价且不占 DSP：
    //     nk == 1 -> cb          + k
    //     nk == 9 -> (cb << 3) + cb + k
    wire [19:0] cb_sh3 = {cb, 3'b0};              // cb * 8
    wire [19:0] cb_x9  = cb_sh3 + {4'b0, cb};     // cb * 9
    wire [KW-1:0]      widx = ksize_r ? (cb + {12'b0, k})
                                      : (cb_x9 + {16'b0, k});"""


def main():
    txt = SRC.read_text(encoding='utf-8')
    if OLD not in txt:
        raise SystemExit('[mk_v10] 未找到目标行，v9 源码是否已改动？')
    txt = txt.replace(OLD, NEW)
    txt = txt.replace(
        '// v9 ', '// v10', 1) if '// v9 ' in txt else txt
    DST.write_text(txt, encoding='utf-8')
    print(f'[mk_v10] wrote {DST}  ({len(txt.splitlines())} lines)')


if __name__ == '__main__':
    main()
