`timescale 1ns/1ps
// TPU DMA：DDR <-> tpu_conv_engine 的流式搬运（激活 / 输出 / 权重）
// ---------------------------------------------------------------------------
// 定位：tpu_mmio 只负责「CPU 逐 beat 搬」，一个 32x32 特征图要几百次 MMIO
//       访问，功能通了但性能没法用。本模块把激活输入 / 卷积输出 / 权重
//       改成 DDR 直连。
//
// 传输单位（与 Pango native AXI 口一致）：
//   - 地址宽 28 位，单位是 4 字节字，且必须 8 字（32 字节）对齐
//     => axi_addr = {byte_addr[29:5], 3'b000}
//   - 每次 AR/AW 固定传 32 字节（arlen/awlen 恒 0，仲裁器串行占用通道）
//
// 读侧（DDR -> 引擎激活口）：
//   256 位 beat 进 FIFO，按 C 字节切片喂引擎。FIFO 解耦 AR 往返延迟与
//   引擎的 act_ready 背压，避免每 4 个 beat 就等一次 DDR 往返。
// 写侧（引擎输出 -> DDR）：
//   P 字节的 out beat 攒满 32 字节后发一次 AW+W。P=16 时 2 拍攒 1 次写，
//   与 arbiter 的 AW(1拍)+W(1拍) 恰好同速，不会憋死引擎。
//
// 权重侧（DDR -> 引擎权重口，wt_en=1 时启用）：
//   **独占第二条 AXI 读口**，与激活口并行，在下游仲裁器处逐 beat 交错。
//   为什么值得独立：引擎的 fill_act = S_WLOAD || S_COMP —— 装载权重的
//   同时也在填行缓冲。共用一条口时权重优先会把激活完全饿死，装完权重进
//   S_COMP 后行缓冲是空的，还得再干等（实测 act=113）。独立后两条流交错，
//   装载与填充重叠，实测 570 -> 490 拍。
//   一次装载 = 一个输出通道组（P 个 lane）的全部权重，字节布局约定：
//       [cout][cb][k]  每单元 C 字节
//     => 组内线性序 = lane p 外层、idx=cb*nk+k 内层，与引擎的
//        w_addr = p*WDEPTH + idx 完全对应，无需除法，用两个计数器即可。
//   灌完最后一个切片时拉高一拍 w_loaded，引擎据此置 wbank_ready 并推进。
//
// 约束：C 与 P 必须 <= 32 且能整除 32（当前配置 C=8、P=16 均满足）。
// ---------------------------------------------------------------------------
module tpu_dma #(
    parameter integer C          = 8,
    parameter integer P          = 16,
    parameter integer WDEPTH     = 512,     // 权重 bank 深度（须为 2 的幂）
    parameter integer FIFO_DEPTH = 4,       // 读 FIFO 的 256-bit beat 数
    // 权重读口模式：
    //   0 = 独占第二条 AXI 读口（默认，实测更快）
    //   1 = 与激活共用一条 AXI 读口，权重优先
    // 实测（4x4x8->32，两条口经同一仲裁器汇聚、同一 DDR 从设备）：
    //   共用 570 拍（comp=288 act=113 starve=291 wload=152）
    //   独立 490 拍（comp=288 act=  1 starve=165 wload=185）
    // 直觉上「让权重尽快装完」应该更快，实测相反：权重独占期间激活一次都
    // 读不到，装完进 S_COMP 后行缓冲是空的，引擎得再干等 113 拍；交错虽然
    // 让 wload 多了 33 拍，但激活早已就位，净省 80 拍（-14%）。
    parameter integer WT_SHARED_PORT = 0
) (
    input  wire        clk, resetn,

    // ---- 控制（来自 tpu_mmio）----
    input  wire        start,             // 1 拍启动脉冲，与引擎 start 同拍
    input  wire        dma_en,            // 本次运行走 DMA（0 则本模块完全静默）
    input  wire        act_hold,          // 1 = 暂缓喂激活（等首组权重 w_loaded）
    input  wire [31:0] act_addr,          // 激活源字节地址（需 32 字节对齐）
    input  wire [31:0] act_bytes,         // 激活总字节数（需为 C 的倍数）
    input  wire [15:0] act_rounds,        // 激活要重复喂几遍 = cout/P（输出通道组数）
    input  wire [31:0] out_addr,          // 输出目的字节地址（需 32 字节对齐）
    input  wire [31:0] out_bytes,         // 输出总字节数（需为 P 的倍数）
    // ---- 输出布局重塑（多层串联用）----
    // 引擎吐出的顺序是 [pg][row][col][P]（pg 最外层）。线性写回 DDR 的话下一层
    // 没法直接用：下一层按 [row][cb][col][C] 消费，cb 正是上一层的 pg。
    // out_seg != 0 时启用分段写，把同一份数据流写成 [row][pg][col][P]，
    // 正好是下一层（cb=pg、C=P）期望的布局，于是多层可以直接串起来。
    //   out_seg  = W_out * P            每个 (pg,row) 行段的字节数，须为 32 的倍数
    //   out_rowj = (npg-1) * W_out * P  写完一个行段后调到同 pg 下一行的增量
    //   out_rows = H_out
    // out_seg == 0 -> 退回原来的线性连续写（单层 / 完全向后兼容）。
    input  wire [31:0] out_seg,
    input  wire [31:0] out_rowj,
    input  wire [15:0] out_rows,
    output wire        busy,
    output wire        done,
    output wire [7:0]  error_code,

    // ---- 引擎 激活侧（本模块为主）----
    output wire            act_valid,
    input  wire            act_ready,
    output wire [C*8-1:0]  act_data,

    // ---- 引擎 输出侧（本模块为从）----
    input  wire            out_valid,
    output wire            out_ready,
    input  wire [P*8-1:0]  out_data,

    // ---- 引擎 权重侧（wt_en=1 时本模块为主）----
    input  wire            wt_en,             // 权重也走 DMA（0 = CPU 经 MMIO 搬）
    input  wire [31:0]     wt_addr,           // 权重区字节基址（需 32 字节对齐）
    input  wire [31:0]     wt_group_bytes,    // 每通道组的字节数 = P*ncb*nk*C
    input  wire [15:0]     wt_lane_slices,    // 每 lane 的切片数 = ncb*nk
    input  wire [15:0]     wt_group_slices,   // 每通道组的切片数 = P*ncb*nk
    input  wire            pg_load_req,       // 引擎：请把 load_pg 组灌进 load_bank
    input  wire [15:0]     load_pg,           // 引擎：待装载的组号
    output wire            w_valid,
    input  wire            w_ready,
    output wire [15:0]     w_addr,
    output wire [C*8-1:0]  w_data,
    output wire            w_loaded,          // 1 拍脉冲：本组已灌完

    // ---- AXI 读口 #1：激活（接 preprocess_ddr_read_arbiter 的 pp 侧）----
    output wire [27:0]  araddr,
    output wire         arvalid,
    input  wire         arready,
    input  wire         rvalid, rlast,
    input  wire [3:0]   rid,
    input  wire [255:0] rdata,
    input  wire         read_error,       // 上游仲裁器 protocol_error

    // ---- AXI 读口 #2：权重（独立通道，见文件头「权重读口为什么要独立」）----
    // 与激活口完全对等：各有自己的 AR/R 握手，互不阻塞。
    // 若系统只提供一条读通道，把两组口接到同一个 arbiter 的 pp 侧亦可
    // （等价于旧的权重优先模式），功能不变、只是退回串行。
    output wire [27:0]  wt_araddr,
    output wire         wt_arvalid,
    input  wire         wt_arready,
    input  wire         wt_rvalid, wt_rlast,
    input  wire [3:0]   wt_rid,
    input  wire [255:0] wt_rdata,
    input  wire         wt_read_error,    // 权重口所属仲裁器 protocol_error

    // ---- AXI 写（接 camera_ddr_write_arbiter 的 cam 侧）----
    output wire [27:0]  awaddr,
    output wire         awvalid,
    input  wire         awready,
    output wire [255:0] wdata,
    input  wire         wready
);
    // ---------------- 常量 ----------------
    localparam ACT_BPB = 32 / C;          // 每 256-bit beat 含多少个 act beat
    localparam OUT_BPW = 32 / P;          // 多少个 out beat 凑一次 32 字节写
    localparam IDXW    = (ACT_BPB > 1) ? $clog2(ACT_BPB)  : 1;
    localparam WCNTW   = (OUT_BPW > 1) ? $clog2(OUT_BPW)  : 1;
    localparam FAW     = (FIFO_DEPTH > 1) ? $clog2(FIFO_DEPTH) : 1;
    localparam KW      = (WDEPTH <= 2) ? 1 : $clog2(WDEPTH);   // 权重 bank 地址位宽

    localparam W_IDLE = 2'd0, W_AW = 2'd1, W_W = 2'd2;

    // ---------------- 激活读 FIFO ----------------
    reg [255:0] mem [0:FIFO_DEPTH-1];
    reg [FAW-1:0]  wptr, rptr;
    reg [FAW:0]    count;                 // 多一位以表示 "满"
    wire fifo_full  = (count == FIFO_DEPTH[FAW:0]);
    wire fifo_empty = (count == 0);
    wire [255:0] fifo_rd = mem[rptr];

    // ---------------- 权重读 FIFO ----------------
    reg [255:0] wmem [0:FIFO_DEPTH-1];
    reg [FAW-1:0]  wwptr, wrptr;
    reg [FAW:0]    wcount;
    wire wfifo_full  = (wcount == FIFO_DEPTH[FAW:0]);
    wire wfifo_empty = (wcount == 0);
    wire [255:0] wfifo_rd = wmem[wrptr];

    // ---------------- 激活读控制 ----------------
    reg         active, ar_sent;
    reg [31:0]  r_bytes_left;             // 还需从 DDR 读的字节数（32 对齐）
    reg [31:0]  a_beats_left;             // 还需喂给引擎的 act beat 数
    reg [31:0]  cur_ar_byte;              // 下一个 AR 的字节地址
    reg [IDXW-1:0] aidx;                  // 当前 beat 内的切片序号
    reg [15:0]  rounds_left;              // 激活还需重复喂的遍数

    // ---------------- 权重读控制 ----------------
    reg         wt_busy, wt_ar_sent, wt_loaded_r;
    reg [15:0]  wt_last_pg;               // 上一轮已装载的组号
    reg         wt_have_pg;                // 本轮是否已装载过至少一组
    reg [15:0]  wt_pg_r;                   // 本次装载锁存的组号
    reg [31:0]  wt_bytes_left;             // 本组还需从 DDR 读的字节数
    reg [31:0]  wt_cur_byte;
    reg [15:0]  wt_slices_left;            // 本组还需灌给引擎的切片数
    reg [15:0]  lane_p;                    // 当前 lane（0..P-1）
    reg [KW-1:0] widx;                     // 当前 lane 内的 idx（0..ncb*nk-1）
    reg [IDXW-1:0] waidx;                  // 权重 beat 内的切片序号
    reg         r_owner;                   // 共用模式：在飞的 AR 归属 1=权重

    // ---------------- 写控制 ----------------
    reg [1:0]    wstate;
    reg [31:0]   o_beats_left;            // 还需从引擎收的 out beat 数
    reg [31:0]   cur_aw_byte;
    reg [255:0]  wbuf;
    reg [WCNTW-1:0] wcnt;
    reg          wfull;                   // wbuf 已攒满，待写出

    // ---------------- 输出分段写控制（out_seg != 0 时生效）----------------
    reg [31:0]   seg_left;                // 当前行段还剩多少字节要写
    reg [15:0]   row_left;                 // 当前 pg 还剩多少行
    reg [31:0]   pgbase;                  // 当前 pg 段的首地址
    wire        seg_mode = (out_seg != 0);

    // ---------------- 读口仲裁 ----------------
    wire act_want = active && !ar_sent && !fifo_full && (r_bytes_left != 0);
    wire wt_want  = wt_busy && !wt_ar_sent && !wfifo_full && (wt_bytes_left != 0);
    wire shared   = (WT_SHARED_PORT != 0);
    // 共用模式下权重优先：引擎等在权重上，激活晚一拍只是行缓冲填得慢。
    wire ar_fire  = (act_want || (shared && wt_want)) && arready;

    assign arvalid    = act_want || (shared && wt_want);
    assign araddr     = (shared && wt_want) ? {wt_cur_byte[29:5], 3'b000}
                                            : {cur_ar_byte[29:5], 3'b000};
    assign wt_arvalid = !shared && wt_want;
    assign wt_araddr  = {wt_cur_byte[29:5], 3'b000};

    assign awvalid = (wstate == W_AW);
    assign awaddr  = {cur_aw_byte[29:5], 3'b000};
    assign wdata   = wbuf;

    // ---------------- 引擎侧握手 ----------------
    // act_hold 必须在模块内部生效：若只在外层把 act_valid 拉低，本模块仍会
    // 用自己的 act_valid 去递减 a_beats_left，计数会凭空跑飞。
    assign act_valid = active && !fifo_empty && (a_beats_left != 0) && !act_hold;
    assign act_data  = fifo_rd[aidx*C*8 +: C*8];
    // wbuf 满时必须反压：否则会丢 out beat（引擎无重发）
    assign out_ready = active && !wfull && (o_beats_left != 0);

    // 权重口：引擎 w_ready 恒 1，但仍按标准握手写，便于以后加反压
    assign w_valid = wt_busy && !wfifo_empty && (wt_slices_left != 0);
    assign w_data  = wfifo_rd[waidx*C*8 +: C*8];
    // w_addr = lane_p * WDEPTH + widx；WDEPTH 是 2 的幂故用移位。
    // lane_p <= P-1 <= 31、KW <= 9 -> 31<<9 = 15872，不会溢出 16 位。
    assign w_addr  = (lane_p << KW) + {{(16-KW){1'b0}}, widx};
    assign w_loaded = wt_loaded_r;

    // ---------------- 分段写的下一地址 ----------------
    // 段内顺序推进 32 字节；段末换行跳 out_rowj；行末换 pg 回到 pgbase+out_seg。
    wire [31:0] aw_plus32    = cur_aw_byte + 32'd32;
    wire        seg_done     = seg_mode && (seg_left <= 32'd32);
    wire        row_last     = (row_left == 16'd1);
    wire [31:0] next_aw      = (!seg_mode || !seg_done) ? aw_plus32
                             : (!row_last)             ? (aw_plus32 + out_rowj)
                                                        : (pgbase + out_seg);
    wire [31:0] next_seg_left = !seg_done ? (seg_left - 32'd32) : out_seg;
    wire [15:0] next_row_left = !seg_done ? row_left
                              : (!row_last) ? (row_left - 16'd1) : out_rows;
    wire [31:0] next_pgbase   = (!seg_done || !row_last) ? pgbase
                                                         : (pgbase + out_seg);

    wire act_fire = act_valid && act_ready;
    wire out_fire = out_valid && out_ready;
    wire wt_fire  = w_valid && w_ready;
    wire act_push = active && ar_sent && rvalid && !(shared && r_owner);
    wire wt_push  = wt_busy && wt_ar_sent &&
                    (shared ? (rvalid && r_owner) : wt_rvalid);
    wire act_pop  = act_fire && (aidx == ACT_BPB - 1);         // 激活 FIFO 出队
    wire wt_pop   = wt_fire && (waidx == ACT_BPB - 1);         // 权重 FIFO 出队

    // ---------------- 完成判定 ----------------
    wire rd_done = active && (r_bytes_left == 0) && (a_beats_left == 0) &&
                   fifo_empty && (rounds_left == 0);
    wire wr_done = active && (o_beats_left == 0) && !wfull && (wstate == W_IDLE);
    wire all_done = rd_done && wr_done && !wt_busy;

    reg done_r;
    reg [7:0] err_r;
    assign busy       = active || wt_busy;
    assign done       = done_r;
    assign error_code = err_r;

    // =======================================================================
    integer i;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            active <= 0; ar_sent <= 0; aidx <= 0;
            r_bytes_left <= 0; a_beats_left <= 0; cur_ar_byte <= 0;
            rounds_left <= 0;
            wstate <= W_IDLE; o_beats_left <= 0; cur_aw_byte <= 0;
            wbuf <= 0; wcnt <= 0; wfull <= 0;
            seg_left <= 0; row_left <= 0; pgbase <= 0;
            wptr <= 0; rptr <= 0; count <= 0;
            wwptr <= 0; wrptr <= 0; wcount <= 0;
            wt_busy <= 0; wt_ar_sent <= 0; wt_loaded_r <= 0;
            wt_have_pg <= 0; wt_last_pg <= 0;
            wt_pg_r <= 0; wt_bytes_left <= 0; wt_cur_byte <= 0;
            wt_slices_left <= 0; lane_p <= 0; widx <= 0; waidx <= 0;
            r_owner <= 0;
            done_r <= 0; err_r <= 0;
            for (i = 0; i < FIFO_DEPTH; i = i + 1) begin
                mem[i]  <= 0;
                wmem[i] <= 0;
            end
        end else begin
            done_r <= 0;
            wt_loaded_r <= 0;                       // w_loaded 只拉高一拍

            // ---- 启动 ----
            if (start && dma_en) begin
                active       <= 1'b1;
                ar_sent      <= 1'b0;
                aidx         <= 0;
                wptr         <= 0; rptr <= 0; count <= 0;
                // 读：向上取整到 32 字节，避免尾拍漏读
                r_bytes_left <= ((act_bytes + 31) >> 5) << 5;
                a_beats_left <= act_bytes / C;
                cur_ar_byte  <= act_addr;
                // 引擎循环是 pg 外层：每换一个输出通道组，行缓冲都要从头
                // 重填一遍，因此激活必须被喂 npg 遍（CPU 模式下是 CPU 重喂）。
                rounds_left  <= (act_rounds > 16'd1) ? (act_rounds - 16'd1) : 16'd0;
                // 写
                o_beats_left <= out_bytes / P;
                cur_aw_byte  <= out_addr;
                wstate       <= W_IDLE;
                wcnt         <= 0;
                wfull        <= 1'b0;
                seg_left     <= out_seg;
                row_left     <= out_rows;
                pgbase       <= out_addr;
                done_r       <= 1'b0;
                err_r        <= 0;
                // 权重：装载由 pg_load_req 的上升沿触发（首组也在内），
                // 这里只做清空，避免上一层的残留请求被误当成新请求。
                wt_busy      <= 1'b0;
                wt_ar_sent   <= 1'b0;
                wt_have_pg   <= 1'b0;
                wwptr <= 0; wrptr <= 0; wcount <= 0; waidx <= 0;
            end

            // ---- 激活：一遍喂完还有遍数 -> 回到 act_addr 重读 ----
            if (active && (a_beats_left == 0) && fifo_empty &&
                (r_bytes_left == 0) && (rounds_left != 0)) begin
                a_beats_left  <= act_bytes / C;
                r_bytes_left  <= ((act_bytes + 31) >> 5) << 5;
                cur_ar_byte   <= act_addr;
                wptr <= 0; rptr <= 0; count <= 0;
                rounds_left   <= rounds_left - 16'd1;
            end

            // ---- 权重：新的一组请求 -> 启动装载 ----
            // 两个坑都踩过，这里一次说清：
            //  [1] 不能用 pg_load_req 的上升沿：引擎在算第 pg 组时就把 pg+1 的请求
            //     挂上了，本组装完请求仍是高，只认上升沿则第二组永远不装载。
            //  [2] 也不能只认电平：w_loaded 发出后，引擎要隔一拍才置 wbank_ready，
            //     这一拍里请求仍高且 load_pg 还没变，会把同一组重复装载一遍
            //     （实测表现为切片计数从 6 反弹回 116，并覆盖正在计算的 bank）。
            //  因此判据是「请求有效 && 空闲 && 组号变了（或本轮还没装载过）」。
            if (wt_en && !wt_busy && pg_load_req &&
                (!wt_have_pg || (load_pg != wt_last_pg))) begin
                wt_busy        <= 1'b1;
                wt_ar_sent     <= 1'b0;
                wt_pg_r        <= load_pg;
                wt_last_pg     <= load_pg;
                wt_have_pg     <= 1'b1;
                // 组基址 = wt_addr + load_pg * wt_group_bytes（装载瞬间算一次，
                // 结果锁存，不在关键路径上，综合器爱用 DSP 就用）
                wt_cur_byte    <= wt_addr + load_pg * wt_group_bytes;
                wt_bytes_left  <= wt_group_bytes;
                wt_slices_left <= wt_group_slices;
                lane_p         <= 0;
                widx           <= 0;
                waidx          <= 0;
                wwptr          <= 0; wrptr <= 0; wcount <= 0;
            end

            // ---- 读：发地址 ----
            // shared 模式下 AR 归属必须在发出的那一刻锁进 r_owner（AR 不可撤销）；
            // 独立模式下两条口各握各的手，**绝不能**用激活口的 arready 去置
            // wt_ar_sent —— 否则权重会以为 AR 已发出而永远等不到自己的 rvalid。
            if (shared) begin
                if (ar_fire) begin
                    r_owner <= wt_want;
                    if (wt_want) wt_ar_sent <= 1'b1;
                    else         ar_sent    <= 1'b1;
                end
            end else begin
                if (act_want && arready)   ar_sent    <= 1'b1;
                if (wt_want && wt_arready) wt_ar_sent <= 1'b1;
            end

            // ---- 读：收数据（按归属入各自的 FIFO）----
            if (act_push) begin
                mem[wptr]   <= rdata;
                wptr        <= (wptr == FIFO_DEPTH - 1) ? 0 : wptr + 1'b1;
                ar_sent     <= 1'b0;
                cur_ar_byte <= cur_ar_byte + 32;
                r_bytes_left<= (r_bytes_left > 32) ? (r_bytes_left - 32) : 0;
            end
            if (wt_push) begin
                wmem[wwptr]  <= wt_rdata;
                wwptr        <= (wwptr == FIFO_DEPTH - 1) ? 0 : wwptr + 1'b1;
                wt_ar_sent   <= 1'b0;
                wt_cur_byte  <= wt_cur_byte + 32;
                wt_bytes_left<= (wt_bytes_left > 32) ? (wt_bytes_left - 32) : 0;
            end

            // count / wcount 必须单点赋值：同拍 push+pop 时若各写一次非阻塞
            // 赋值，两者都拿旧值计算，后写的会覆盖前者，计数凭空少 1。
            if (act_push || act_pop) count  <= count  + act_push - act_pop;
            if (wt_push  || wt_pop)  wcount <= wcount + wt_push  - wt_pop;

            // ---- 读：喂引擎激活 ----
            if (act_fire) begin
                a_beats_left <= a_beats_left - 1'b1;
                if (aidx == ACT_BPB - 1) begin
                    aidx  <= 0;
                    rptr  <= (rptr == FIFO_DEPTH - 1) ? 0 : rptr + 1'b1;
                end else begin
                    aidx <= aidx + 1'b1;
                end
            end

            // ---- 权重：灌引擎 ----
            if (wt_fire) begin
                wt_slices_left <= wt_slices_left - 1'b1;
                // idx 走到 lane 末尾 -> 换 lane；组内线性序 [p][idx] 与
                // w_addr = p*WDEPTH + idx 一一对应，所以不需要除法
                if (widx == wt_lane_slices[KW-1:0] - 1'b1) begin
                    widx   <= 0;
                    lane_p <= lane_p + 1'b1;
                end else begin
                    widx <= widx + 1'b1;
                end
                if (waidx == ACT_BPB - 1) begin
                    waidx <= 0;
                    wrptr <= (wrptr == FIFO_DEPTH - 1) ? 0 : wrptr + 1'b1;
                end else begin
                    waidx <= waidx + 1'b1;
                end
            end

            // ---- 权重：本组灌完 -> 拉 w_loaded ----
            if (wt_busy && (wt_slices_left == 0)) begin
                wt_busy     <= 1'b0;
                wt_loaded_r <= 1'b1;
            end

            // ---- 写：收引擎输出 ----
            if (out_fire) begin
                wbuf         [wcnt*P*8 +: P*8] <= out_data;
                o_beats_left <= o_beats_left - 1'b1;
                if (wcnt == OUT_BPW - 1) begin
                    wcnt  <= 0;
                    wfull <= 1'b1;
                end else begin
                    wcnt <= wcnt + 1'b1;
                end
            end

            // ---- 写：推 DDR ----
            case (wstate)
                W_IDLE: if (wfull) wstate <= W_AW;
                W_AW:   if (awready) begin
                            if (wready) begin
                                wfull <= 0; wstate <= W_IDLE;
                                cur_aw_byte <= next_aw;
                                if (seg_mode) begin
                                    seg_left <= next_seg_left;
                                    row_left <= next_row_left;
                                    pgbase   <= next_pgbase;
                                end
                            end else begin
                                wstate <= W_W;
                            end
                        end
                W_W:    if (wready) begin
                            wfull <= 0; wstate <= W_IDLE;
                            cur_aw_byte <= next_aw;
                            if (seg_mode) begin
                                seg_left <= next_seg_left;
                                row_left <= next_row_left;
                                pgbase   <= next_pgbase;
                            end
                        end
                default: wstate <= W_IDLE;
            endcase

            // ---- 错误 ----
            if ((active || wt_busy) && (read_error || (!shared && wt_read_error)) && err_r == 0)
                err_r <= 8'd5;

            // ---- 收尾 ----
            if (all_done) begin
                active <= 1'b0;
                done_r <= 1'b1;
            end
        end
    end

    // 参数合法性（仅仿真期提示，综合期会被忽略）
    // synthesis translate_off
    initial begin
        if (C > 32 || (32 % C) != 0)
            $display("[tpu_dma] ERROR: C=%0d 必须 <=32 且整除 32", C);
        if (P > 32 || (32 % P) != 0)
            $display("[tpu_dma] ERROR: P=%0d 必须 <=32 且整除 32", P);
        if (WDEPTH != (1 << KW))
            $display("[tpu_dma] ERROR: WDEPTH=%0d 必须是 2 的幂（w_addr 用移位生成）", WDEPTH);
    end
    // synthesis translate_on

endmodule
