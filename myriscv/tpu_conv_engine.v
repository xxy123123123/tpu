`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// TPU 卷积引擎 v9  ——  参数化 P(输出通道并行) x C(输入通道并行) INT8 MAC 阵列
//                       综合化改造：同步 BRAM + 像素级流水
//
// 相对 v4 的改造（动机：P32C16 实测综合 LUTRAM 49152/39500 超限 124%，DRM=0）
//   [1] 存储阵列全部改为"平铺 + 同步读"，让综合器推断 DRM(BRAM)：
//       wmem[P][2*WDEPTH]   权重：{bank, widx} 平铺，每 p 一块双端口 DRM
//       lmem[NBUF*NCBMAX*W]  行缓冲：{bank, cb, col} 平铺，双端口（写=填充，读=计算）
//       消掉的 acc[P][W] 数组是 v4 最大的 LUTRAM 消耗源之一（见 [2]）
//   [2] 循环重排 pg -> oy -> ox(列) -> cb -> k（v4 是 cb 外层、ox 内层）：
//       每个输出像素的全部 ncb*nk 个项在 P 个 32bit 像素累加寄存器 pacc 里
//       一次算完 -> 末项最终值交 S3 requant -> 发射。于是：
//         - acc 整行缓冲数组不再存在（262Kbit LUTRAM -> 32 个寄存器）
//         - S_DRAIN 状态不再存在（发射即像素完成，天然与计算重叠）
//         - v4 的 emit_ox 尾随发射机制全部删除
//   [3] 三级流水（全局只有输出背压一种冻结）：
//       S0(发射级)  控制寄存器 -> 组合算地址/行就绪 -> 打入 S1
//       S1(数据级)  lmem/wmem 同步读输出 -> 寄存器，供 S2 组合乘法
//       S2(乘加级)  组合乘法 + C 项加法树 -> pacc 累加 / s2_acc（末项最终值）
//       S3(发射级)  RHA requant（33 位幅值运算）+ 输出寄存器
//       S2/S3 拆分动机：合级时关键路径 = 乘法 + 加法树 + requant 幅值运算，
//       P32C16 综合估计仅 ~55MHz；拆开两级各担一半，冲刺 200MHz。
//       row_ready 未就绪 = 插入气泡（S1 无效拍），不冻结下游；
//       输出背压(out_valid&&!out_ready) = 全局冻结（保发射寄存器不被覆盖）
//
// 沿用 v2/v3/v4 的全部机制（权重乒乓、5-bank 内容寻址行缓冲、后台填充、
// 1x1 卷积、越界补零、RHA requant），对外接口与 beat 顺序完全不变。
//
// 行缓冲 bank 映射（内容寻址，无模算术）
//   写侧: 行 vr -> bank fbank_cnt（0..NBUF-1 自由轮转，随 vr 同步递增/复位）
//   读侧: bank = 命中 bvalid[i] && brow[i]==want_row 的那个 i（至多一个）
//   覆盖写安全 = 旧行号 < 当前窗口最老行（brow[fbank_cnt] < row_base）
//   同步读安全性：brow/bvalid 与该行最后一个 lmem 写同拍更新，而发射级
//   看到就绪至少晚一拍 -> 读数据必然已写入（BRAM 读旧值语义无害）
//
// 数据流（循环嵌套，从外到内）
//   pg(输出通道组) -> oy(输出行) -> ox(输出列) -> cb(输入通道块) -> k(核位)
//   填充指针 vr 全局单调递增，与状态机解耦
//
// 吞吐：满流水时每周期 P*C 次 INT8 MAC，峰值 = P*C*freq
// -----------------------------------------------------------------------------
module tpu_conv_engine #(
    parameter P        = 16,    // 输出通道并行度
    parameter C        = 8,     // 输入通道并行度
    parameter MAX_W    = 256,   // 输入宽度上限
    parameter NCBMAX   = 8,     // 行缓冲的通道块维度上限（需 >= cin/C）
    parameter WDEPTH   = 512,   // 权重 bank 深度 = ceil(CIN/C)*9 上限（需 2 的幂）
    parameter MAX_COUT = 256,   // bias 表深度
    parameter NBUF     = 5      // 行缓冲 bank 数：窗口3行 + stride2 预取2行
) (
    input  wire clk,
    input  wire resetn,

    // ---- 层配置 ----
    input  wire        cfg_valid,
    output wire        cfg_ready,
    input  wire [15:0] cfg_h,
    input  wire [15:0] cfg_w,
    input  wire [15:0] cfg_cin,
    input  wire [15:0] cfg_cout,
    input  wire [1:0]  cfg_stride,   // 1 或 2
    input  wire        cfg_pad,      // 0 或 1
    input  wire        cfg_ksize,    // 0: 3x3, 1: 1x1
    input  wire        cfg_relu,     // 1: 输出钳到 0..127
    input  wire [4:0]  cfg_shift,    // requant 右移量

    // ---- 权重写入: 每 beat 写 C 个字节（同一 k 的 C 个输入通道）----
    // 物理 bank 由 load_bank 指定，w_addr 只给 {逻辑 bank p, idx}
    input  wire            w_valid,
    output wire            w_ready,
    input  wire [15:0]     w_addr,   // = p*WDEPTH + idx, idx = cb*NK + k
    input  wire [C*8-1:0]  w_data,

    // ---- bias 写入 ----
    input  wire              b_valid,
    output wire              b_ready,
    input  wire [7:0]        b_addr,
    input  wire signed [31:0] b_data,

    // ---- 激活流: 每 beat C 个通道的同一像素 ----
    input  wire            act_valid,
    output wire            act_ready,
    input  wire [C*8-1:0]  act_data,

    // ---- 输出流: 每 beat P 个通道（像素完成即发射）----
    output wire            out_valid,
    input  wire            out_ready,
    output wire [P*8-1:0]  out_data,

    // ---- 权重组预取握手（乒乓）----
    // pg_load_req=1 时请把 load_pg 组的权重灌进 load_bank，灌完拉高 w_loaded。
    // 计算第 pg 组期间会请求 pg+1 组（写到另一 bank），因此除首组外可与计算重叠。
    input  wire        w_loaded,
    output wire [15:0] load_pg,
    output wire        load_bank,
    output wire        pg_load_req,
    output wire [15:0] cur_pg,

    output wire busy,
    output wire done,

    // ---- stall 分类计数器（上板定位瓶颈用）----
    // 前五个是互斥的 bucket，覆盖全部 busy 周期：
    //   total = comp + act + out + wload + drain
    // starve 不是 bucket，而是正交标记：引擎想填行缓冲但上游没给数据。
    // 它可能落在 wload 里（装载权重的同时在填行缓冲，两条流抢同一条 AXI
    // 读口），也可能落在 act 里。所以 starve 大 => 查 DDR 读带宽/仲裁，
    // 而不是简单地去怪权重。
    output wire [31:0] perf_total,    // busy 总周期
    output wire [31:0] perf_comp,    // 真在发射/计算
    output wire [31:0] perf_act,     // 计算态但行缓冲未就绪（等激活）
    output wire [31:0] perf_starve,  // 想填行缓冲但上游没给数据（act 的子集）
    output wire [31:0] perf_out,     // 输出背压（下游没取走）
    output wire [31:0] perf_wload,   // 等权重装载
    output wire [31:0] perf_drain    // 末项已发射，等流水排空
);
    localparam CW  = (MAX_W   <= 2) ? 1 : $clog2(MAX_W);
    localparam KW  = (WDEPTH  <= 2) ? 1 : $clog2(WDEPTH);
    localparam CBM = (NCBMAX  <= 2) ? 1 : $clog2(NCBMAX);
    localparam BNK = (NBUF    <= 2) ? 1 : $clog2(NBUF);

    localparam S_IDLE  = 3'd0;
    localparam S_WLOAD = 3'd1;
    localparam S_COMP  = 3'd2;
    localparam S_DONE  = 3'd4;   // 3'd3 保留给 v4 的 S_DRAIN（测试台统计口径兼容）

    // ---------------- 存储（同步读，推断 DRM）----------------
    // wmem 见下方 generate（依赖 widx，必须在其声明之后）
    // lmem: 行缓冲，地址 {bank, cb, col}
    reg [C*8-1:0]     lmem [0:NBUF*NCBMAX*MAX_W-1];
    // bias 表：按输出通道组切片成宽字（P 路并排），只需 1 个读口。
    // 若写成 bias[0:MAX_COUT-1] 再 P 路并行读，PDS 会把整个阵列复制 P 份
    // （实测 P=32 时复制 32 份 256x32，白吃 32 块 DRM18K）。
    localparam BDEPTH = (MAX_COUT + P - 1) / P;      // 组数（P/MAX_COUT 均为 2 的幂时即 MAX_COUT/P）
    localparam BDW    = (BDEPTH  <= 2) ? 1 : $clog2(BDEPTH);
    reg [P*32-1:0] biasw [0:BDEPTH-1];
    wire [15:0] bidx = b_addr / P;                   // 组内偏移
    wire [15:0] bmod = b_addr % P;                   // lane 号
    wire [P*32-1:0] bias_rd = biasw[pg[BDW-1:0]];
    // 当前 pg 的 bias 快照（每 pg 装载一次，避免发射级读 bias 阵列）
    reg [P*32-1:0]  bias_p;

    // ---------------- 配置锁存 ----------------
    reg [15:0] h_r, w_r, cin_r, cout_r, hout_r, wout_r;
    reg [1:0]  stride_r;
    reg        pad_r, ksize_r, relu_r;
    reg [4:0]  shift_r;
    reg [15:0] ncb_r;    // 输入通道块数 = cin/C
    reg [15:0] npg_r;    // 输出通道组数 = cout/P

    // ---------------- 控制状态（S0 发射级）----------------
    reg [2:0]  state;
    reg [15:0] pg, oy, ox, cb;
    reg [3:0]  k;
    reg        ctrl_done;           // 本 pg 末项已发射，控制冻结等流水排空

    // ---------------- 填充（后台，与 v4 相同）----------------
    reg signed [16:0] vr;           // 下一个待装载的虚拟输入行
    reg [15:0] fcb, fx;             // 填充的块计数 / 列计数
    reg [BNK-1:0]     fbank_cnt;    // 写侧 bank 轮转计数器
    reg signed [16:0] brow [0:NBUF-1];
    reg [NBUF-1:0]   bvalid;
    reg [1:0]  wbank_ready;

    integer pi, ci, bi2;
    reg signed [31:0] acc_sum;      // 累加中间变量（模块级：PDS 不允许块内声明）

    // ---------------- 握手 ----------------
    assign cfg_ready = (state == S_IDLE);
    assign w_ready   = 1'b1;
    assign b_ready   = 1'b1;
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);
    assign cur_pg    = pg;

    // k 终值与每块权重字数（1x1: k 只跑 1 拍、每 cb 1 个权重字）
    wire [3:0]  klast = ksize_r ? 4'd0 : 4'd8;
    wire [15:0] nk    = ksize_r ? 16'd1 : 16'd9;
    wire signed [16:0] kmax = ksize_r ? 17'sd0 : 17'sd2;   // 窗口最高行偏移

    // ---------------- 输出尺寸（组合）----------------
    wire [16:0] khw  = cfg_ksize ? 17'd1 : 17'd3;
    wire [16:0] ext_h = {1'b0, cfg_h} + (cfg_pad ? 17'd2 : 17'd0) - khw;
    wire [16:0] ext_w = {1'b0, cfg_w} + (cfg_pad ? 17'd2 : 17'd0) - khw;

    // ---------------- 配置合法性检查（防非法参数死锁）----------------
    // cin/C 与 cout/P 必须整除（整数除法，不整除会丢通道甚至死锁：cin<C 时
    // ncb_r=0，last_term_i = (cb==ncb_r-1) 变成 cb==16'hffff 永不成立）。
    // ext_h/ext_w 必须 >= 0（否则 17 位有符号下溢成巨大值，引擎无限计算）。
    // 非法配置直接进 S_DONE 空转，不进入计算，避免挂死整条流水。
    wire cin_bad  = (cfg_cin < C) || ((cfg_cin % C) != 0);
    // cout > MAX_COUT 时偏置表 biasw[BDEPTH] 写越界（b_addr 只有 8 bit），必须拦下
    wire cout_bad = (cfg_cout < P) || ((cfg_cout % P) != 0) || (cfg_cout > MAX_COUT);
    wire h_bad    = $signed(ext_h) < 17'sd0;
    wire w_bad    = $signed(ext_w) < 17'sd0;
    wire cfg_bad  = cin_bad || cout_bad || h_bad || w_bad;

    // synthesis translate_off
    // 仿真期显式报错，方便定位是哪个参数不对
    always @(posedge clk) begin
        if (cfg_valid && cfg_ready) begin
            if (cin_bad)
                $display("[tpu_conv_engine] ERROR: cin=%0d 必须为 C=%0d 的正整数倍", cfg_cin, C);
            if (cout_bad)
                $display("[tpu_conv_engine] ERROR: cout=%0d 必须为 P=%0d 的正整数倍且 <=MAX_COUT=%0d", cfg_cout, P, MAX_COUT);
            if (h_bad)
                $display("[tpu_conv_engine] ERROR: h=%0d 太小（pad/ksize 下溢）", cfg_h);
            if (w_bad)
                $display("[tpu_conv_engine] ERROR: w=%0d 太小（pad/ksize 下溢）", cfg_w);
        end
    end
    // synthesis translate_on

    // ---------------- 权重预取请求 ----------------
    wire [15:0] next_pg_w = (pg >= npg_r - 16'd1) ? 16'd0 : (pg + 16'd1);
    wire        has_next  = (state != S_IDLE) && (state != S_DONE) && (pg < npg_r - 16'd1);

    assign load_pg     = (state == S_WLOAD) ? pg : next_pg_w;
    assign load_bank   = (state == S_WLOAD) ? pg[0] : next_pg_w[0];
    assign pg_load_req = (state == S_WLOAD) ? !wbank_ready[pg[0]]
                                            : (has_next && !wbank_ready[next_pg_w[0]]);

    // ---------------- 权重预取完成检测（上升沿）----------------
    reg w_loaded_d;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) w_loaded_d <= 1'b0;
        else         w_loaded_d <= w_loaded;
    end
    wire w_loaded_pe = w_loaded && !w_loaded_d;

    // ---------------- S0: 窗口定位 / 行就绪（内容寻址）----------------
    // stride 只有 1/2 两种取值：用「移位 or 直连」代替乘法。
    // 原写法被综合器映射到 APM(DSP)，仅 tco 就 2.65ns，是 v7 布局后关键路径的起点。
    wire [16:0] oy_x2 = {oy, 1'b0};
    wire signed [16:0] row_base = $signed((stride_r == 2'd2) ? oy_x2 : {1'b0, oy})
                                - (pad_r ? 17'sd1 : 17'sd0);
    wire [3:0] ky = ksize_r ? 4'd0 : (k >= 4'd6) ? 4'd2 : (k >= 4'd3) ? 4'd1 : 4'd0;
    wire signed [16:0] want_row = row_base + $signed({13'b0, ky});
    // 窗口越界行（padding 行）本身视为就绪：数据当零，无需等待
    wire want_real = (want_row >= 0) && (want_row < $signed({1'b0, h_r}));

    wire [NBUF-1:0] bank_hit;
    genvar gb;
    generate
        for (gb = 0; gb < NBUF; gb = gb + 1) begin : g_bh
            assign bank_hit[gb] = want_real && bvalid[gb] && (brow[gb] == want_row);
        end
    endgenerate
    wire row_ready = !want_real || (|bank_hit);

    reg [BNK-1:0] rbank_idx;
    always @* begin
        rbank_idx = {BNK{1'b0}};
        for (bi2 = 0; bi2 < NBUF; bi2 = bi2 + 1)
            if (bank_hit[bi2]) rbank_idx = bi2[BNK-1:0];
    end

    wire [3:0] kx = ksize_r ? 4'd0
                  : (k - ((ky == 4'd2) ? 4'd6 : (ky == 4'd1) ? 4'd3 : 4'd0));
    wire [16:0] ox_x2 = {ox, 1'b0};
    wire signed [16:0] col_rd = $signed((stride_r == 2'd2) ? ox_x2 : {1'b0, ox})
                              + $signed({13'b0, kx}) - (pad_r ? 17'sd1 : 17'sd0);
    wire col_ok = (col_rd >= 0) && (col_rd < $signed({1'b0, w_r}));
    wire [CW-1:0] col_idx = col_ok ? col_rd[CW-1:0] : {CW{1'b0}};
    // 行越界（want_real=0）或列越界（col_ok=0）都补零
    wire win_ok = col_ok && want_real;

    // ---------------- S0: 读地址（组合，发射拍打入 S1）----------------
    // ---------------- v10: 权重地址去掉 APM 乘法 ----------------
    // v10布局后关键路径就是它：cb*nk 被映射到 APM（td 2.754ns），输出再以
    // fanout=32 广播到所有 wmem bank（net 3.139ns），两级逻辑吃掉 10.7ns。
    // nk 只有 1 / 9 两种取值，用常数移位-加乘法替代，完全等价且不占 DSP：
    //     nk == 1 -> cb          + k
    //     nk == 9 -> (cb << 3) + cb + k
    wire [19:0] cb_sh3 = {cb, 3'b0};              // cb * 8
    wire [19:0] cb_x9  = cb_sh3 + {4'b0, cb};     // cb * 9
    wire [KW-1:0]      widx = ksize_r ? (cb + {12'b0, k})
                                      : (cb_x9 + {16'b0, k});
    wire [BNK+CBM+CW-1:0] lbuf_raddr = {rbank_idx, cb[CBM-1:0], col_idx};

    // ---------------- 权重存储（每 lane 一块独立 1D RAM，generate 作用域）----------------
    // PDS 2022.2 综合器 RAM 推断结论（探针实测）：
    //   [OK]     generate 作用域 1D 数组 + 同步读（读在 posedge 里）→ 推断 DRM（BRAM）
    //   [LUTRAM] generate 作用域 1D 数组 + 异步读（continuous assign）→ 退化分布式 RAM
    //   [崩溃]   模块级 2D 数组 + always 内 for 循环按第一维写 → rtl-infer 段错误
    //   [FF 堆]  模块级 2D 数组 + generate 分 lane 写 → 不推断 RAM，退化百万 FF
    // 因此 wmem 必须用"generate 作用域 + 同步读"这一种写法。
    genvar gp;

    // ---------------- 发射级控制信号 ----------------
    wire first_term_i = (cb == 16'd0)   && (k == 4'd0);
    wire last_term_i  = (cb == ncb_r - 16'd1) && (k == klast);
    wire fin_pix_i    = last_term_i && (ox == wout_r - 16'd1) && (oy == hout_r - 16'd1);
    wire fin_layer_i  = fin_pix_i  && (pg == npg_r - 16'd1);

    // ---------------- S2/S3 流水数据级 ----------------
    reg              s1_valid;
    reg              s1_first, s1_last, s1_zero, s1_pgfin, s1_fin;
    reg [15:0]       s1_chbase;               // 发射时的 pg*P（requant 通道判断用）
    reg [C*8-1:0]    s1_act;                  // lmem 同步读数据
    reg [P*C*8-1:0]  s1_w;                    // wmem 同步读数据（所有 lane 一次读齐）
    reg signed [31:0] pacc [0:P-1];           // 像素累加寄存器（替代 acc 数组）

    // S2a（乘积拍）控制伴随信号：与 pr 同步延迟一拍
    reg              s2a_valid, s2a_first, s2a_last, s2a_pgfin, s2a_fin;
    reg [15:0]       s2a_chbase;
    // S2b（加法树前半拍）控制伴随信号：与 tm 同步再延迟一拍
    reg              s2b_valid, s2b_first, s2b_last, s2b_pgfin, s2b_fin;
    reg [15:0]       s2b_chbase;

    // S3（requant 级）数据
    reg              s2_valid, s2_fin, s2_pgfin;
    reg [15:0]       s2_chbase;
    reg [P*32-1:0]   s2_acc;                  // 每像素最终累加值（requant 输入）

    // 输出发射寄存器
    reg              out_valid_r, out_fin_r, out_pgfin_r;
    reg [P*8-1:0]    out_data_r;

    assign out_valid = out_valid_r;
    assign out_data  = out_data_r;

    // 全局冻结：输出背压（唯一需要停整条流水的条件）
    wire gfreeze = out_valid_r && !out_ready;
    // 发射条件：计算态 + 本组还有项 + 不冻结 + 行就绪
    wire issue = (state == S_COMP) && !ctrl_done && !gfreeze && row_ready;

    // ---------------- wmem：generate 作用域 per-lane RAM（同步读 → DRM）----------------
    generate
        for (gp = 0; gp < P; gp = gp + 1) begin : g_wmem
            reg [C*8-1:0] wbank [0:2*WDEPTH-1];
            // 写口：主机按 w_addr = p*WDEPTH + idx 灌权重，bank 由 load_bank 指定
            always @(posedge clk)
                if (w_valid && (w_addr[15:KW] == gp))
                    wbank[{load_bank, w_addr[KW-1:0]}] <= w_data;
            // 读口：无条件同步读（generate 作用域里条件读会退化成分布式 RAM——PDS 推断
            // 规律实测：generate+门控读→LUTRAM，generate+无条件读→DRM）。
            // 不需要门控：{pg[0], widx} 只随 issue 推进，gfreeze 期间地址稳定，
            // 重复读同一地址无害；s1_w 只在 s1_valid=1 时被消费。
            always @(posedge clk)
                s1_w[gp*C*8 +: C*8] <= wbank[{pg[0], widx}];
        end
    endgenerate

    // 发射完成事件（供状态机迁移判断）
    wire emit_done  = out_valid_r && out_ready && out_fin_r;
    wire pg_switch  = out_valid_r && out_ready && out_pgfin_r && !out_fin_r
                    && (state == S_COMP);

    // ---------------- 后台填充（与权重加载/计算并行）----------------
    wire [BNK-1:0] fbank      = fbank_cnt;
    wire        fill_act   = (state == S_WLOAD) || (state == S_COMP);
    // 覆盖写安全：目标 bank 的旧行必须比当前窗口最老行还老（或从未装载）
    wire        fill_bank_ok = !bvalid[fbank] || (brow[fbank] < row_base);
    // 需求上限：只预取到"下一输出行窗口最高行"，避免超前浪费带宽
    wire        fill_limit_ok = (vr <= row_base + kmax + $signed({14'b0, stride_r}));
    wire        vr_real  = (vr >= 0) && (vr < $signed({1'b0, h_r}));
    wire        fill_go  = fill_act && fill_bank_ok && fill_limit_ok;

    assign act_ready = fill_go && vr_real;

    wire [BNK-1:0] fbank_wrap = (fbank_cnt == NBUF-1) ? {BNK{1'b0}}
                                                       : fbank_cnt + {{(BNK-1){1'b0}}, 1'b1};

    // 乘法阵列（S1 寄存器 -> 组合乘法 -> S2 拍累加）
    wire [C*8-1:0] act_eff = s1_zero ? {C*8{1'b0}} : s1_act;
    wire signed [15:0] prod [0:P*C-1];
    genvar mp, mc;
    generate
        for (mp = 0; mp < P; mp = mp + 1) begin : g_mp
            for (mc = 0; mc < C; mc = mc + 1) begin : g_mc
                assign prod[mp*C+mc] = $signed(act_eff[mc*8 +: 8])
                                     * $signed(s1_w[(mp*C+mc)*8 +: 8]);
            end
        end
    endgenerate

    // ---------------- S2a: 乘积寄存器（切断 乘法 -> 加法树 的组合路径）----------------
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

    // ---------------- S2: C 项乘积归约（显式加法树 + 中间打拍）----------------
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

    // requant（RHA 舍入；不用 $signed——iverilog 对 function 内 $signed 有兼容问题）
    function [7:0] tpu_requant;
        input [31:0] a;
        input [4:0]  shift;
        input        relu;
        reg neg;
        reg [32:0] mag, rnd, sc, ssc;
        begin
            neg = a[31];
            mag = neg ? (~{a[31], a} + 33'd1) : {a[31], a};
            rnd = (shift == 5'd0) ? mag : mag + (33'd1 << (shift - 5'd1));
            sc  = rnd >> shift;
            ssc = neg ? (~sc + 33'd1) : sc;
            if (relu) begin
                if (neg || sc == 33'd0)     tpu_requant = 8'd0;
                else if (sc > 33'd127)      tpu_requant = 8'd127;
                else                        tpu_requant = ssc[7:0];
            end else                        tpu_requant = ssc[7:0];
        end
    endfunction

    // ---------------- 主状态机（S0 控制 + 状态迁移）----------------
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            state <= S_IDLE;
            pg <= 0; oy <= 0; ox <= 0; cb <= 0; k <= 0;
            ctrl_done <= 0;
            wbank_ready <= 0;
        end else begin
            // 权重预取完成置位（单一 always 原则；与消费清零同拍冲突时消费优先）
            if (w_loaded_pe)
                wbank_ready[load_bank] <= 1'b1;
            case (state)
                S_IDLE: if (cfg_valid) begin
                    h_r      <= cfg_h;
                    w_r      <= cfg_w;
                    cin_r    <= cfg_cin;
                    cout_r   <= cfg_cout;
                    stride_r <= cfg_stride;
                    pad_r    <= cfg_pad;
                    ksize_r  <= cfg_ksize;
                    relu_r   <= cfg_relu;
                    shift_r  <= cfg_shift;
                    hout_r   <= (cfg_stride == 2'd2) ? (ext_h >> 1) + 16'd1
                                                     : ext_h[15:0] + 16'd1;
                    wout_r   <= (cfg_stride == 2'd2) ? (ext_w >> 1) + 16'd1
                                                     : ext_w[15:0] + 16'd1;
                    ncb_r    <= cfg_cin / C;
                    npg_r    <= cfg_cout / P;
                    pg       <= 0;
                    oy       <= 0;
                    ctrl_done <= 0;
                    wbank_ready <= 2'b00;
                    // 非法配置不进计算（否则 cin<C 时 ncb_r=0 导致死锁）
                    state    <= cfg_bad ? S_DONE : S_WLOAD;
                end

                // 等当前组的权重就位；就位后清标记并快照 bias
                S_WLOAD: if (wbank_ready[pg[0]]) begin
                    wbank_ready[pg[0]] <= 1'b0;
                    cb <= 0; ox <= 0; oy <= 0; k <= 0; ctrl_done <= 0;
                    bias_p <= bias_rd;                  // 整组 P 路一次快照（单读口）
                    state <= S_COMP;
                end

                // 计算：issue 有效时推进 cb/k/ox/oy（pg -> oy -> ox -> cb -> k）
                S_COMP: begin
                    if (emit_done) begin
                        // 本层最后一个像素发射完成
                        state <= S_DONE;
                    end else if (pg_switch) begin
                        // 本 pg 最后一个像素发射完成：切组
                        pg <= pg + 16'd1;
                        cb <= 0; ox <= 0; oy <= 0; k <= 0; ctrl_done <= 0;
                        state <= S_WLOAD;
                    end else if (issue) begin
                        if (k == klast) begin
                            k <= 0;
                            if (cb == ncb_r - 16'd1) begin
                                cb <= 0;
                                if (ox == wout_r - 16'd1) begin
                                    ox <= 0;
                                    if (oy == hout_r - 16'd1) begin
                                        oy <= 0;
                                        ctrl_done <= 1;       // 本组末项已发射
                                    end else oy <= oy + 16'd1;
                                end else ox <= ox + 16'd1;
                            end else cb <= cb + 16'd1;
                        end else k <= k + 4'd1;
                    end
                end

                S_DONE: state <= S_IDLE;
                default: ;
            endcase
        end
    end

    // ---------------- S1: 数据级捕获（同步读 lmem/wmem）----------------
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            s1_valid <= 1'b0;
            s1_first <= 1'b0; s1_last <= 1'b0; s1_zero <= 1'b0;
            s1_pgfin <= 1'b0; s1_fin <= 1'b0; s1_chbase <= 16'd0;
        end else if (!gfreeze) begin
            s1_valid <= issue;
            if (issue) begin
                s1_first  <= first_term_i;
                s1_last   <= last_term_i;
                s1_zero   <= !win_ok;
                s1_pgfin  <= fin_pix_i;
                s1_fin    <= fin_layer_i;
                s1_chbase <= pg * P;
            end
            s1_act <= lmem[lbuf_raddr];
        end
    end

    // ---------------- S2: 乘加（组合乘法 + 加法树 -> pacc / s2_acc）----------------
    // 关键路径止于 pacc/s2_acc 寄存器；requant 的 33 位幅值运算下沉到 S3，
    // 两级拆分后 Fmax 可各砍一半左右（综合实测 v5 合级约 55MHz，布局前估计）。
    // ---- S2a: 控制伴随信号与 pr 同步延迟一拍 ----
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

    // ---- S2b: 加法树前半伴随信号（与 tm 同步延迟一拍）----
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

    // ---- S2b: 加法树归约 + 像素累加 -> pacc / s2_acc ----
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            s2_valid <= 1'b0; s2_fin <= 1'b0; s2_pgfin <= 1'b0; s2_chbase <= 16'd0;
        end else if (!gfreeze) begin
            // s2_valid 每拍重写（一拍进一拍出，S3 每拍消费一个）；
            // 背压时本级冻结，s2_valid 保持，恢复后由 S3 补消费
            s2_valid <= s2b_valid && s2b_last;
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
            end
        end
    end

    // ---------------- S3: requant + 发射（独立一级，关键路径仅幅值运算）----------------
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            out_valid_r <= 1'b0; out_fin_r <= 1'b0; out_pgfin_r <= 1'b0;
        end else if (!gfreeze) begin
            if (out_valid_r && out_ready)
                out_valid_r <= 1'b0;          // 消费（后写覆盖：同拍新像素优先）
            if (s2_valid) begin
                for (pi = 0; pi < P; pi = pi + 1)
                    out_data_r[pi*8 +: 8] <= ((s2_chbase + pi) < cout_r)
                                           ? tpu_requant(s2_acc[pi*32 +: 32],
                                                         shift_r, relu_r)
                                           : 8'd0;
                out_valid_r <= 1'b1;
                out_fin_r   <= s2_fin;
                out_pgfin_r <= s2_pgfin;
            end
        end
    end

    // ---------------- 后台填充（vr/fcb/fx/fbank_cnt/brow/bvalid 只在这里写）----
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            vr <= 0; fcb <= 0; fx <= 0; fbank_cnt <= 0;
            bvalid <= {NBUF{1'b0}};
            for (pi = 0; pi < NBUF; pi = pi + 1) brow[pi] <= 17'sd0;
        end else if (cfg_valid && cfg_ready) begin
            // 新层：行缓冲整体作废（bvalid 清零即可，brow 会被覆盖写）
            vr <= cfg_pad ? -17'sd1 : 17'sd0;
            fcb <= 0; fx <= 0; fbank_cnt <= 0; bvalid <= {NBUF{1'b0}};
        end else if (pg_switch) begin
            // pg 切换：下一输出通道组从头重填行缓冲
            vr <= pad_r ? -17'sd1 : 17'sd0;
            fcb <= 0; fx <= 0; fbank_cnt <= 0; bvalid <= {NBUF{1'b0}};
        end else if (fill_go) begin
            if (!vr_real) begin
                // padding 行：不消耗输入带宽，标记无效后推进
                bvalid[fbank] <= 1'b0;
                brow[fbank]   <= vr;
                vr            <= vr + 17'sd1;
                fbank_cnt     <= fbank_wrap;
            end else if (act_valid) begin
                lmem[{fbank, fcb[CBM-1:0], fx[CW-1:0]}] <= act_data;
                if (fx == w_r - 16'd1) begin
                    fx <= 0;
                    if (fcb == ncb_r - 16'd1) begin
                        fcb <= 0;
                        bvalid[fbank] <= 1'b1;
                        brow[fbank]   <= vr;
                        vr            <= vr + 17'sd1;
                        fbank_cnt     <= fbank_wrap;
                    end else fcb <= fcb + 16'd1;
                end else fx <= fx + 16'd1;
            end
        end
    end

    // ---------------- 权重 / bias 写入（wmem 各 lane 在 generate 内自写）----------------
    always @(posedge clk) begin
        if (b_valid)
            for (pi = 0; pi < P; pi = pi + 1)
                if (bmod == pi) biasw[bidx[BDW-1:0]][pi*32 +: 32] <= b_data;
    end

    // =======================================================================
    // stall 分类计数器
    // 口径互斥且完备（覆盖所有 busy 周期）；每次新层 cfg_valid 时清零。
    // =======================================================================
    reg [31:0] c_total, c_comp, c_act, c_starve, c_out, c_wload, c_drain;

    assign perf_total  = c_total;
    assign perf_comp   = c_comp;
    assign perf_act    = c_act;
    assign perf_starve = c_starve;
    assign perf_out    = c_out;
    assign perf_wload  = c_wload;
    assign perf_drain  = c_drain;

    wire running  = (state != S_IDLE);
    wire st_wload = (state == S_WLOAD);
    wire st_out   = running && gfreeze;
    wire st_drain = running && !gfreeze &&
                    ((state == S_DONE) || ((state == S_COMP) && ctrl_done));
    wire st_act   = (state == S_COMP) && !ctrl_done && !gfreeze && !row_ready;
    wire st_comp  = (state == S_COMP) && !ctrl_done && !gfreeze &&  row_ready;
    // 引擎明确要填（act_ready）但上游没给数据 -> 欠喂，是 act 的子集
    wire st_starve = act_ready && !act_valid;

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            c_total <= 0; c_comp <= 0; c_act <= 0; c_starve <= 0;
            c_out <= 0; c_wload <= 0; c_drain <= 0;
        end else if (cfg_valid && (state == S_IDLE)) begin
            // 新一层开始：清零，计数从 S_WLOAD 那一拍起算
            c_total <= 0; c_comp <= 0; c_act <= 0; c_starve <= 0;
            c_out <= 0; c_wload <= 0; c_drain <= 0;
        end else if (running) begin
            c_total <= c_total + 32'd1;
            if (st_wload)  c_wload  <= c_wload  + 32'd1;
            if (st_out)    c_out    <= c_out    + 32'd1;
            if (st_drain)  c_drain  <= c_drain  + 32'd1;
            if (st_act)    c_act    <= c_act    + 32'd1;
            if (st_comp)   c_comp   <= c_comp   + 32'd1;
            if (st_starve) c_starve <= c_starve + 32'd1;
        end
    end

endmodule
