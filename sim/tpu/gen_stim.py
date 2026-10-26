"""TPU 卷积引擎激励生成 + bit-exact 参考模型。

生成的激励顺序严格对齐 RTL 的循环嵌套：
    激活流  单遍：虚拟行 vr 从 -pad 单调递增到 (hout-1)*stride-pad+kmax+stride，
           越界行（vr<0 或 vr>=H）跳过不占 beat，真实行按 (cb, x) 全通道块读。
           行缓冲跨输出通道组保留 -> 多个 pg 也只读这一遍。
    权重    pg 内 p -> cb -> k(0..NK-1)，一字 C 字节
    输出    pg -> oy -> ox(0..Wout-1)，每 beat P 字节

参考模型的 requant 语义与 RTL（cnn_requantize 同款 RHA）逐位一致：
    mag = |acc|; rounded = mag + (1 << (shift-1)); scaled = rounded >> shift
    relu: 负数或 0 -> 0，>127 -> 127；否则按符号还原
"""
import argparse
import json
from pathlib import Path

import numpy as np


def requant(v: int, shift: int, relu: bool) -> int:
    neg = v < 0
    mag = -v if neg else v
    rnd = mag if shift == 0 else mag + (1 << (shift - 1))
    sc = rnd >> shift
    if relu:
        if neg or sc == 0:
            return 0
        return 127 if sc > 127 else sc
    return -sc if neg else sc


def conv_ref(act, wgt, bias, stride, pad, shift, relu, ksize=3):
    """act (CIN,H,W) int8, wgt (COUT,CIN,KH,KW) int8, bias (COUT,) int32
    -> out (COUT,Hout,Wout) int32（已 requant）"""
    cin, h, w = act.shape
    cout = wgt.shape[0]
    hout = (h + 2 * pad - ksize) // stride + 1
    wout = (w + 2 * pad - ksize) // stride + 1
    out = np.zeros((cout, hout, wout), dtype=np.int64)
    for co in range(cout):
        for oy in range(hout):
            for ox in range(wout):
                acc = int(bias[co])
                for ci in range(cin):
                    for ky in range(ksize):
                        row = oy * stride + ky - pad
                        if row < 0 or row >= h:
                            continue
                        for kx in range(ksize):
                            col = ox * stride + kx - pad
                            if col < 0 or col >= w:
                                continue
                            acc += int(act[ci, row, col]) * int(wgt[co, ci, ky, kx])
                out[co, oy, ox] = requant(acc, shift, relu)
    return out


def pack_beats(vals, nbytes):
    """把一串 0..255 的字节按 nbytes 打包成整数列表（低字节在前 = 通道 0）"""
    out = []
    for i in range(0, len(vals), nbytes):
        word = 0
        for j in range(nbytes):
            if i + j < len(vals):
                word |= (int(vals[i + j]) & 0xFF) << (8 * j)
        out.append(word)
    return out


def hex_lines(words, width_bytes):
    return "".join(f"{w:0{width_bytes * 2}x}\n" for w in words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--P', type=int, default=4)
    ap.add_argument('--C', type=int, default=4)
    ap.add_argument('--H', type=int, default=8)
    ap.add_argument('--W', type=int, default=8)
    ap.add_argument('--CIN', type=int, default=8)
    ap.add_argument('--COUT', type=int, default=8)
    ap.add_argument('--stride', type=int, default=1, choices=[1, 2])
    ap.add_argument('--pad', type=int, default=1, choices=[0, 1])
    ap.add_argument('--ksize', type=int, default=3, choices=[1, 3])
    ap.add_argument('--shift', type=int, default=6)
    ap.add_argument('--relu', type=int, default=1)
    ap.add_argument('--seed', type=int, default=20261022)
    ap.add_argument('--out', default='sim/tpu/stim')
    ap.add_argument('--max-w', type=int, default=256)
    ap.add_argument('--ncbmax', type=int, default=8)
    ap.add_argument('--wdepth', type=int, default=512)
    ap.add_argument('--max-cout', type=int, default=256)
    a = ap.parse_args()

    assert a.CIN % a.C == 0, f'CIN({a.CIN}) 必须是 C({a.C}) 的整数倍'
    assert a.COUT % a.P == 0, f'COUT({a.COUT}) 必须是 P({a.P}) 的整数倍'
    assert a.W <= a.max_w, f'W({a.W}) 超过 MAX_W({a.max_w})'
    assert a.CIN // a.C <= a.ncbmax, f'ncb({a.CIN // a.C}) 超过 NCBMAX({a.ncbmax})'

    rng = np.random.default_rng(a.seed)
    act = rng.integers(-128, 127, (a.CIN, a.H, a.W), dtype=np.int64)
    wgt = rng.integers(-128, 127, (a.COUT, a.CIN, a.ksize, a.ksize), dtype=np.int64)
    bias = rng.integers(-2000, 2000, a.COUT, dtype=np.int64)

    out = conv_ref(act, wgt, bias, a.stride, a.pad, a.shift, a.relu, a.ksize)
    hout, wout = out.shape[1], out.shape[2]
    ncb = a.CIN // a.C
    npg = a.COUT // a.P
    kmax = a.ksize - 1
    nk = a.ksize * a.ksize

    # ---------- 激活流（v3：每个 pg 单遍，填充与计算并行）----------
    # 填充指针 vr 从 -pad 单调递增到 final_limit（最后一个输出行窗口最高行 + stride
    # 的预取超前量）。padding 行（vr<0 或 vr>=H）不产生 beat；真实行按 (cb, x) 全
    # 通道块读。RTL 的填充与计算并行，但消费顺序恒为 vr 递增。
    final_limit = (hout - 1) * a.stride - a.pad + kmax + a.stride
    act_beats = []
    for _pg in range(npg):
        for vr in range(-a.pad, final_limit + 1):
            if 0 <= vr < a.H:
                for cb in range(ncb):
                    for x in range(a.W):
                        act_beats.extend(int(act[cb * a.C + c, vr, x]) & 0xFF
                                         for c in range(a.C))

    # 理论一遍数据量（beat）：单个 pg 输入特征图恰好读一遍
    nact_theory = a.H * a.W * ncb
    # v1 的对照量：每个 (oy, cb) 都重填 3 行，且每个 pg 重读整层。仅用于带宽对比。
    nact_v1 = 0
    for _pg in range(npg):
        for oy in range(hout):
            for cb in range(ncb):
                for b in range(a.ksize):
                    row = oy * a.stride + b - a.pad
                    if 0 <= row < a.H:
                        nact_v1 += a.W

    # ---------- 权重（每 pg 一份，只装有效字数 ncb*nk）----------
    nw = ncb * nk                   # 每 bank 有效字数
    assert nw <= a.wdepth, f'有效权重字数 {nw} 超 WDEPTH({a.wdepth})'
    # 布局: 索引 pg*P*NW + p*NW + idx, idx = cb*nk + k
    wgt_words = []
    for pg in range(npg):
        for p in range(a.P):
            for idx in range(nw):
                cb, k = idx // nk, idx % nk
                ky, kx = k // a.ksize, k % a.ksize
                wgt_words.extend(int(wgt[pg * a.P + p, cb * a.C + c, ky, kx]) & 0xFF
                                 for c in range(a.C))

    # ---------- 期望输出 ----------
    exp_beats = []
    for pg in range(npg):
        for oy in range(hout):
            for ox in range(wout):
                exp_beats.extend(int(out[pg * a.P + p, oy, ox]) & 0xFF
                                 for p in range(a.P))

    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'act.hex').write_text(hex_lines(pack_beats(act_beats, a.C), a.C), encoding='ascii')
    (dst / 'wgt.hex').write_text(hex_lines(pack_beats(wgt_words, a.C), a.C), encoding='ascii')
    (dst / 'bias.hex').write_text(hex_lines([int(b) & 0xFFFFFFFF for b in bias], 4),
                                  encoding='ascii')
    (dst / 'exp.hex').write_text(hex_lines(pack_beats(exp_beats, a.P), a.P), encoding='ascii')

    cfg = dict(P=a.P, C=a.C, H=a.H, W=a.W, CIN=a.CIN, COUT=a.COUT,
               stride=a.stride, pad=a.pad, ksize=a.ksize, shift=a.shift, relu=a.relu,
               hout=hout, wout=wout, ncb=ncb, npg=npg,
               nact=len(pack_beats(act_beats, a.C)),
               nexp=len(pack_beats(exp_beats, a.P)),
               wdepth=a.wdepth, max_w=a.max_w, max_cout=a.max_cout,
               ncbmax=a.ncbmax, nact_v1=nact_v1, nact_theory=nact_theory, nw=nw)
    (dst / 'cfg.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')

    # 供测试台 include 的参数头
    vh = "\n".join([
        f"localparam TB_P       = {a.P};",
        f"localparam TB_C       = {a.C};",
        f"localparam TB_H       = {a.H};",
        f"localparam TB_W       = {a.W};",
        f"localparam TB_CIN     = {a.CIN};",
        f"localparam TB_COUT    = {a.COUT};",
        f"localparam TB_STRIDE  = {a.stride};",
        f"localparam TB_PAD     = {a.pad};",
        f"localparam TB_KSIZE   = {a.ksize};",
        f"localparam TB_SHIFT   = {a.shift};",
        f"localparam TB_RELU    = {a.relu};",
        f"localparam TB_HOUT    = {hout};",
        f"localparam TB_WOUT    = {wout};",
        f"localparam TB_NCB     = {ncb};",
        f"localparam TB_NPG     = {npg};",
        f"localparam TB_NACT    = {len(pack_beats(act_beats, a.C))};",
        f"localparam TB_NACT_V1 = {nact_v1};",
        f"localparam TB_NACT_TH = {nact_theory};",
        f"localparam TB_NEXP    = {len(pack_beats(exp_beats, a.P))};",
        f"localparam TB_MAX_W   = {a.max_w};",
        f"localparam TB_NCBMAX  = {a.ncbmax};",
        f"localparam TB_WDEPTH  = {a.wdepth};",
        f"localparam TB_NW      = {nw};",
        f"localparam TB_MAX_COUT= {a.max_cout};",
        "",
    ])
    (dst / 'params.vh').write_text(vh, encoding='ascii')

    print(json.dumps(dict(stage='stim', **cfg)))


if __name__ == '__main__':
    main()
