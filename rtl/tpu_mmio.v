`timescale 1ns/1ps
// TPU MMIO 从设备（ABI: tpu_int8_v1）
// ---------------------------------------------------------------------------
// 窗口：0x1000_5000，256 字节，word 对齐（由 data_bus_interconnect 译码）
// 职责：把 CPU 的 32 位 MMIO 访问翻译成 tpu_conv_engine 的流式握手：
//   配置 cfg_* / 权重 w_* / bias b_* / 激活 act_* / 输出 out_* / 乒乓 w_loaded
// 数据宽度换算（C/P 需为 4 的倍数，若不满足按 (N+3)/4 个字打包）：
//   权重 / 激活：每 beat C 字节 -> ceil(C/4) 次 32 位写
//   输出：        每 beat P 字节 -> ceil(P/4) 次 32 位读
// 背压：任何未完成的引擎握手（cfg/w/b/act）都会拉低 req_ready，
//       由 CPU 侧自然等待，不会丢数据。
//
// 两种数据通路（由 R_DMA_CFG.bit0 选择，start 前设定）：
//   dma_en=0  CPU 搬运：激活走 R_ADATA、输出走 R_ODATA，逐 beat 访问
//   dma_en=1  DMA 搬运：激活/输出直接对 DDR 读写（见 tpu_dma.v）
//             权重仍走 R_WDATA —— 量小，且乒乓预取要按 pg_load_req 逐组响应
// ---------------------------------------------------------------------------
module tpu_mmio #(
    parameter integer P        = 16,
    parameter integer C        = 8,
    parameter integer MAX_W    = 256,
    parameter integer NCBMAX   = 8,
    parameter integer WDEPTH   = 512,
    parameter integer MAX_COUT = 256,
    parameter integer NBUF     = 5,
    parameter integer WT_SHARED_PORT = 0, // 0=权重独占第二条读口（默认，实测快 14%）
    parameter integer CLK_HZ   = 93_750_000
) (
    input  wire        clk, resetn,
    // ---- MMIO 从设备口 ----
    input  wire        req_valid, req_write,
    input  wire [1:0]  req_size,
    input  wire [31:0] req_addr, req_wdata,
    input  wire [3:0]  req_wstrb,
    output wire        req_ready,
    output reg         rsp_valid, rsp_error,
    output reg  [31:0] rsp_rdata,
    // ---- DDR 直通口（dma_en=1 时使用）----
    output wire [27:0]  araddr, awaddr,
    output wire         arvalid, awvalid,
    output wire [255:0] wdata,
    input  wire         arready, rvalid, rlast, awready, wready,
    input  wire [3:0]   arid,              // 读通道 ID（本地寄存器索引另用 reg_idx）
    input  wire [255:0] rdata,
    input  wire         read_protocol_error,
    // ---- 权重专用 AXI 读口（与激活口并行，见 tpu_dma 文件头）----
    output wire [27:0]  wt_araddr,
    output wire         wt_arvalid,
    input  wire         wt_arready, wt_rvalid, wt_rlast,
    input  wire [3:0]   wt_rid,
    input  wire [255:0] wt_rdata,
    input  wire         wt_read_protocol_error,
    // ---- 顶层可观测状态（便于接中断 / LED / 调试）----
    output wire        tpu_busy, tpu_done, tpu_pg_req
);
    // ---------------- 寄存器索引（req_addr[7:2]）----------------
    localparam R_ID     = 6'd0;   // R  0x54505531 "TPU1"
    localparam R_CTRL   = 6'd1;   // W  bit0=start bit1=abort bit2=clear
    localparam R_STATUS = 6'd2;   // R  见 STATUS 位定义
    localparam R_CFG_HW = 6'd3;   // W  {h[15:0], w[15:0]}
    localparam R_CFG_CC = 6'd4;   // W  {cin[15:0], cout[15:0]}
    localparam R_CFG_MD = 6'd5;   // W  bit[1:0]=stride bit[2]=pad bit[3]=ksize bit[4]=relu bit[12:8]=shift
    localparam R_WDATA  = 6'd6;   // W  权重字（连续写 ceil(C/4) 次组成一个 beat）
    localparam R_WADDR  = 6'd7;   // W  权重 beat 地址（= p*WDEPTH + idx）
    localparam R_BDATA  = 6'd8;   // W  bias 值（32 位有符号）
    localparam R_BADDR  = 6'd9;   // W  bias 地址
    localparam R_ADATA  = 6'd10;  // W  激活字（连续写 ceil(C/4) 次组成一个 beat）
    localparam R_ODATA  = 6'd11;  // R  输出字（连续读 ceil(P/4) 次取完一个 beat）
    localparam R_LDCTL  = 6'd12;  // R/W 读=乒乓状态，写 bit0 -> w_loaded 脉冲
    localparam R_CYC    = 6'd13;  // R  计算周期计数
    localparam R_ERR    = 6'd14;  // R  错误码
    localparam R_PARAM  = 6'd15;  // R  {P[15:0], C[15:0]}
    // ---- DMA 组（0x40 起）----
    localparam R_DMA_AB = 6'd16;  // W  激活源字节地址（32 字节对齐）
    localparam R_DMA_AL = 6'd17;  // W  激活字节数（C 的倍数）
    localparam R_DMA_OB = 6'd18;  // W  输出目的字节地址（32 字节对齐）
    localparam R_DMA_OL = 6'd19;  // W  输出字节数（P 的倍数）
    localparam R_DMA_CFG= 6'd20;  // W  bit0=dma_en
    localparam R_DMA_ST = 6'd21;  // R  {err[7:0], dma_done, dma_busy, dma_en}
    localparam R_WT_AB  = 6'd22;  // W  权重区字节基址（32 字节对齐）
    localparam R_WT_CFG = 6'd23;  // W  bit0=wt_en（1=权重也走 DMA）
    // 输出布局重塑（多层串联）：0x60/0x64/0x68
    localparam R_OUT_SEG= 6'd24;  // W  每个 (pg,row) 行段字节数 = W_out*P（32 的倍数；0=线性写）
    localparam R_OUT_ROWJ=6'd25;  // W  行段间字节增量 = (npg-1)*W_out*P
    localparam R_OUT_ROWS=6'd26;  // W  H_out
    // stall 分类计数器（只读）
    localparam R_PF_TOT = 6'd27;  // R  0x6C busy 总周期
    localparam R_PF_CMP = 6'd28;  // R  0x70 真在发射/计算
    localparam R_PF_ACT = 6'd29;  // R  0x74 等激活（行缓冲未就绪）
    localparam R_PF_STV = 6'd30;  // R  0x78 上游欠喂（act 的子集）
    localparam R_PF_OUT = 6'd31;  // R  0x7C 等输出背压
    localparam R_PF_WLD = 6'd32;  // R  0x80 等权重装载
    localparam R_PF_DRN = 6'd33;  // R  0x84 等流水排空


    localparam CLW = (C <= 1) ? 1 : $clog2(C);   // C 须为 2 的幂：cin/C 用移位
    localparam PLW = (P <= 1) ? 1 : $clog2(P);   // P 须为 2 的幂：cout/P 用移位
    localparam WB = (C + 3) / 4;          // 权重/激活 每 beat 的 MMIO 字数
    localparam OB = (P + 3) / 4;          // 输出 每 beat 的 MMIO 字数
    localparam WBW = (WB > 1) ? $clog2(WB) : 1;
    localparam OBW = (OB > 1) ? $clog2(OB) : 1;

    // ---------------- 配置暂存 ----------------
    reg [15:0] cfg_h, cfg_w, cfg_cin, cfg_cout;
    reg [1:0]  cfg_stride;
    reg        cfg_pad, cfg_ksize, cfg_relu;
    reg [4:0]  cfg_shift;

    // ---------------- 握手 pending ----------------
    reg cfg_pend, b_pend, w_pend, a_pend;
    reg [C*8-1:0]  w_sh, a_sh;
    reg [WBW-1:0]  w_cnt, a_cnt;
    reg [15:0]     w_addr_cpu, b_addr;      // CPU 搬运模式下的权重/bias 地址
    reg [31:0]     b_sh;

    // 输出缓冲：out_ready 在缓冲空时拉高，引擎输出被自动接住一 beat
    reg [P*8-1:0]  o_sh;
    reg            o_valid;
    reg [OBW-1:0]  o_cnt;

    reg        done, error, w_loaded_r;
    reg [7:0]  error_code;
    reg [31:0] timer, cyc;
    reg        running;               // 已 start、尚未 done

    // ---------------- DMA ----------------
    reg        dma_en;
    reg [31:0] dma_ab, dma_al, dma_ob, dma_ol;
    reg        dma_finished;          // 本次运行的 DMA 已收完
    reg        eng_done_r;            // 引擎 done 是单拍脉冲，必须锁存等 DMA 收尾
    reg        weights_ready;         // 首组权重已 w_loaded（DMA 喂激活的前置条件）
    wire       dma_busy, dma_done;
    wire [7:0] dma_err;
    wire       dma_act_valid, dma_out_ready;
    wire [C*8-1:0] dma_act_data;

    // ---------------- 权重 DMA ----------------
    reg        wt_en;
    reg [31:0] wt_ab;                // 权重区字节基址
    reg [31:0] out_seg_r, out_rowj_r;
    reg [15:0] out_rows_r;
    wire       dma_w_valid, dma_w_loaded;
    wire [15:0] dma_w_addr;
    wire [C*8-1:0] dma_w_data;
    // 权重布局约定 [cout][cb][k]，每单元 C 字节：
    //   每 lane 切片数 = ncb*nk，每通道组切片数 = P*ncb*nk，组字节数再乘 C
    wire [15:0] ncb_w   = cfg_cin >> CLW;
    wire [15:0] nk_w    = cfg_ksize ? 16'd1 : 16'd9;
    wire [31:0] wt_lane_slices32  = ncb_w * nk_w;
    wire [31:0] wt_group_slices32 = wt_lane_slices32 * P;
    wire [15:0] wt_lane_slices    = wt_lane_slices32[15:0];
    wire [15:0] wt_group_slices   = wt_group_slices32[15:0];
    wire [31:0] wt_group_bytes    = wt_group_slices32 << CLW;
    // 输出通道组数 = cout/P：引擎 pg 外层，每换一组激活都要重喂一遍
    wire [15:0] npg_w             = cfg_cout >> PLW;
    wire [15:0] act_rounds_w      = (npg_w < 16'd1) ? 16'd1 : npg_w;

    // ---------------- 引擎信号 ----------------
    wire        cfg_ready, w_ready, b_ready, act_ready;
    wire        act_valid;             // dma_en 选择的激活源
    wire [C*8-1:0] act_data;
    wire        out_ready;             // dma_en 选择的输出汇
    wire        out_valid;
    wire [P*8-1:0] out_data;
    wire        engine_busy, engine_done;
    wire [31:0] perf_total, perf_comp, perf_act, perf_starve,
                perf_out, perf_wload, perf_drain;
    wire [15:0] load_pg, cur_pg;
    wire        load_bank, pg_load_req;
    // 权重口的二选一结果（wt_en 选 DMA / CPU）
    wire        w_valid;
    wire [C*8-1:0] w_data;
    wire [15:0] w_addr;
    wire        w_loaded;

    assign cfg_valid = cfg_pend;
    assign b_valid   = b_pend;
    // 权重源二选一：wt_en=1 由 DMA 从 DDR 搬，wt_en=0 仍是 CPU 经 R_WDATA 灌
    assign w_valid = wt_en ? dma_w_valid : w_pend;
    assign w_data  = wt_en ? dma_w_data  : w_sh;
    assign w_addr  = wt_en ? dma_w_addr  : w_addr_cpu;
    assign w_loaded = wt_en ? dma_w_loaded : w_loaded_r;
    // dma_en=1：激活来自 DMA、输出送往 DMA；否则仍是 CPU 逐 beat 搬运
    //
    // 权重门控是必需的：引擎在等 w_loaded 期间仍会消费 act 填行缓冲，而 CPU 通过
    // MMIO 灌一整组权重（P lane x 9 tap）要上千拍 —— 若放任 DMA 先喂，等权重到位时
    // 行缓冲里的激活早已错位，输出全 0。DMA 的读 FIFO 会先填满然后自然停读，
    // 权重就绪后再开始喂，数据一条不丢。门控信号进 DMA 内部，不是在外层与掉。
    assign act_valid = dma_en ? dma_act_valid : a_pend;
    assign act_data  = dma_en ? dma_act_data  : a_sh;
    assign out_ready = dma_en ? dma_out_ready : !o_valid;

    wire cfg_fire  = cfg_pend  && cfg_ready;
    wire w_fire    = w_pend    && w_ready;
    wire b_fire    = b_pend    && b_ready;
    wire a_fire    = act_valid && act_ready;
    wire o_fire    = out_valid && out_ready;

    // ---------------- MMIO 译码 ----------------
    wire [5:0] rid = req_addr[7:2];
    // wstrb 只约束写：真实 CPU 的读不驱动 wstrb（单元 TB 恒给 4'hf 掩盖了这点）
    wire legal = (req_size == 2'd2) && (req_addr[1:0] == 2'b0) &&
                 (req_addr <= 32'h84) && (!req_write || (req_wstrb == 4'hf));
    wire accept = req_valid && req_ready;
    wire write  = accept && req_write && legal;

    // 未完成握手期间背压 CPU（输出缓冲不算 pending，CPU 轮询 STATUS 即可）
    wire blocked = cfg_pend || b_pend || w_pend || a_pend;
    assign req_ready = resetn && !rsp_valid && !blocked;

    wire start_cmd = write && rid == R_CTRL && req_wdata[0] && !running && !engine_busy;
    wire abort_cmd = write && rid == R_CTRL && req_wdata[1];
    wire clear_cmd = write && rid == R_CTRL && req_wdata[2];
    wire loaded_cmd= write && rid == R_LDCTL && req_wdata[0];

    wire wdata_cmd = write && rid == R_WDATA && !w_pend;
    wire adata_cmd = write && rid == R_ADATA && !a_pend;
    wire bdata_cmd = write && rid == R_BDATA && !b_pend;
    wire waddr_cmd = write && rid == R_WADDR;
    wire baddr_cmd = write && rid == R_BADDR;
    wire cfgw_cmd  = write && (rid == R_CFG_HW || rid == R_CFG_CC || rid == R_CFG_MD);

    wire expired = running && (timer >= CLK_HZ * 3);

    // ---------------- 引擎实例 ----------------
    tpu_conv_engine #(
        .P(P), .C(C), .MAX_W(MAX_W), .NCBMAX(NCBMAX),
        .WDEPTH(WDEPTH), .MAX_COUT(MAX_COUT), .NBUF(NBUF)
    ) u_engine (
        .clk(clk), .resetn(resetn),
        .cfg_valid(cfg_pend), .cfg_ready(cfg_ready),
        .cfg_h(cfg_h), .cfg_w(cfg_w), .cfg_cin(cfg_cin), .cfg_cout(cfg_cout),
        .cfg_stride(cfg_stride), .cfg_pad(cfg_pad), .cfg_ksize(cfg_ksize),
        .cfg_relu(cfg_relu), .cfg_shift(cfg_shift),
        .w_valid(w_valid), .w_ready(w_ready), .w_addr(w_addr), .w_data(w_data),
        .b_valid(b_pend), .b_ready(b_ready), .b_addr(b_addr[7:0]), .b_data(b_sh),
        .act_valid(act_valid), .act_ready(act_ready), .act_data(act_data),
        .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
        .w_loaded(w_loaded), .load_pg(load_pg), .load_bank(load_bank),
        .pg_load_req(pg_load_req), .cur_pg(cur_pg),
        .busy(engine_busy), .done(engine_done),
        .perf_total(perf_total), .perf_comp(perf_comp), .perf_act(perf_act),
        .perf_starve(perf_starve), .perf_out(perf_out),
        .perf_wload(perf_wload), .perf_drain(perf_drain)
    );

    // ---------------- DMA 实例 ----------------
    // start 与引擎 start 同拍；dma_en=0 时模块内部完全静默（不占 AXI 通道）。
    tpu_dma #(.C(C), .P(P), .WDEPTH(WDEPTH), .WT_SHARED_PORT(WT_SHARED_PORT)) u_dma (
        .clk(clk), .resetn(resetn),
        .start(start_cmd), .dma_en(dma_en), .act_hold(!weights_ready),
        .act_addr(dma_ab), .act_bytes(dma_al), .act_rounds(act_rounds_w),
        .out_addr(dma_ob), .out_bytes(dma_ol),
        .out_seg(out_seg_r), .out_rowj(out_rowj_r), .out_rows(out_rows_r),
        .busy(dma_busy), .done(dma_done), .error_code(dma_err),
        .act_valid(dma_act_valid), .act_ready(act_ready), .act_data(dma_act_data),
        .out_valid(out_valid), .out_ready(dma_out_ready), .out_data(out_data),
        .wt_en(wt_en), .wt_addr(wt_ab),
        .wt_group_bytes(wt_group_bytes), .wt_lane_slices(wt_lane_slices),
        .wt_group_slices(wt_group_slices),
        .pg_load_req(pg_load_req), .load_pg(load_pg),
        .w_valid(dma_w_valid), .w_ready(w_ready), .w_addr(dma_w_addr),
        .w_data(dma_w_data), .w_loaded(dma_w_loaded),
        .araddr(araddr), .arvalid(arvalid), .arready(arready),
        .rvalid(rvalid), .rlast(rlast), .rid(arid), .rdata(rdata),
        .read_error(read_protocol_error),
        .wt_araddr(wt_araddr), .wt_arvalid(wt_arvalid), .wt_arready(wt_arready),
        .wt_rvalid(wt_rvalid), .wt_rlast(wt_rlast), .wt_rid(wt_rid),
        .wt_rdata(wt_rdata), .wt_read_error(wt_read_protocol_error),
        .awaddr(awaddr), .awvalid(awvalid), .awready(awready),
        .wdata(wdata), .wready(wready)
    );

    // ---------------- 主状态 ----------------
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            cfg_h <= 0; cfg_w <= 0; cfg_cin <= 0; cfg_cout <= 0;
            cfg_stride <= 0; cfg_pad <= 0; cfg_ksize <= 0; cfg_relu <= 0; cfg_shift <= 0;
            cfg_pend <= 0; b_pend <= 0; w_pend <= 0; a_pend <= 0;
            w_sh <= 0; a_sh <= 0; w_cnt <= 0; a_cnt <= 0;
            w_addr_cpu <= 0; b_addr <= 0; b_sh <= 0;
            o_sh <= 0; o_valid <= 0; o_cnt <= 0;
            done <= 0; error <= 0; error_code <= 0; w_loaded_r <= 0;
            timer <= 0; cyc <= 0; running <= 0;
            dma_en <= 0; dma_ab <= 0; dma_al <= 0; dma_ob <= 0; dma_ol <= 0;
            out_seg_r <= 0; out_rowj_r <= 0; out_rows_r <= 0;
            dma_finished <= 0; eng_done_r <= 0; weights_ready <= 0;
            wt_en <= 0; wt_ab <= 0;
            rsp_valid <= 0; rsp_error <= 0; rsp_rdata <= 0;
        end else begin
            rsp_valid <= 0;
            w_loaded_r <= 0;
            if (cfg_fire)  cfg_pend <= 0;
            if (b_fire)    b_pend <= 0;
            if (w_fire)  begin w_pend <= 0; w_addr_cpu <= w_addr_cpu + 16'd1; end
            if (a_fire)    a_pend <= 0;
            // DMA 模式下输出直接进 DMA 的写缓冲，不进 CPU 可读的 o_sh
            if (o_fire && !dma_en) begin o_sh <= out_data; o_valid <= 1; o_cnt <= 0; end

            if (accept) begin
                rsp_valid <= 1; rsp_error <= 0; rsp_rdata <= 0;
                if (!legal) begin
                    rsp_error <= 1; error_code <= 1;          // 1=非法访问
                end else if (req_write) begin
                    case (rid)
                        R_CTRL:   if (!(start_cmd || abort_cmd || clear_cmd)) begin
                                      rsp_error <= 1; error_code <= 1;
                                  end
                        R_WDATA:  if (!wdata_cmd) begin rsp_error <= 1; error_code <= 1; end
                        R_ADATA:  if (!adata_cmd) begin rsp_error <= 1; error_code <= 1; end
                        R_BDATA:  if (!bdata_cmd) begin rsp_error <= 1; error_code <= 1; end
                        default:  ;
                    endcase
                end else begin
                    case (rid)
                        R_ID:     rsp_rdata <= 32'h54505531;                  // "TPU1"
                        R_STATUS: rsp_rdata <= {23'b0, dma_busy, pg_load_req, o_valid,
                                                a_pend, w_pend, cfg_pend,
                                                error, done, engine_busy};
                        R_LDCTL:  rsp_rdata <= {15'b0, load_bank, load_pg};
                        R_CYC:    rsp_rdata <= cyc;
                        R_ERR:    rsp_rdata <= {24'b0, error_code};
                        R_PARAM:  rsp_rdata <= {P[15:0], C[15:0]};
                        R_DMA_ST: rsp_rdata <= {dma_err, 5'b0, dma_finished, dma_busy, dma_en};
                        R_DMA_CFG:rsp_rdata <= {31'b0, dma_en};
                        R_WT_AB:  rsp_rdata <= wt_ab;
                        R_WT_CFG: rsp_rdata <= {31'b0, wt_en};
                        R_OUT_SEG:  rsp_rdata <= out_seg_r;
                        R_OUT_ROWJ: rsp_rdata <= out_rowj_r;
                        R_OUT_ROWS: rsp_rdata <= {16'b0, out_rows_r};
                        R_PF_TOT: rsp_rdata <= perf_total;
                        R_PF_CMP: rsp_rdata <= perf_comp;
                        R_PF_ACT: rsp_rdata <= perf_act;
                        R_PF_STV: rsp_rdata <= perf_starve;
                        R_PF_OUT: rsp_rdata <= perf_out;
                        R_PF_WLD: rsp_rdata <= perf_wload;
                        R_PF_DRN: rsp_rdata <= perf_drain;
                        R_ODATA:  if (o_valid && !dma_en) begin
                                      rsp_rdata <= o_sh[o_cnt*32 +: 32];
                                      if (o_cnt == OB - 1) begin
                                          o_valid <= 0;
                                      end else begin
                                          o_cnt <= o_cnt + 1'b1;
                                      end
                                  end else begin
                                      rsp_error <= 1; error_code <= 3;        // 3=输出空读
                                  end
                        default:  rsp_rdata <= 0;
                    endcase
                end
            end

            // ---- 写配置 ----
            if (cfgw_cmd) begin
                case (rid)
                    R_CFG_HW: begin cfg_h <= req_wdata[31:16]; cfg_w <= req_wdata[15:0]; end
                    R_CFG_CC: begin cfg_cin <= req_wdata[31:16]; cfg_cout <= req_wdata[15:0]; end
                    R_CFG_MD: begin
                        cfg_stride <= req_wdata[1:0];
                        cfg_pad    <= req_wdata[2];
                        cfg_ksize  <= req_wdata[3];
                        cfg_relu   <= req_wdata[4];
                        cfg_shift  <= req_wdata[12:8];   // 与上面的位域不重叠
                    end
                    default: ;
                endcase
            end

            // ---- 权重 / bias / 激活 数据打包 ----
            if (wdata_cmd) begin
                w_sh[w_cnt*32 +: 32] <= req_wdata;
                if (w_cnt == WB - 1) begin
                    w_cnt <= 0; w_pend <= 1;
                end else begin
                    w_cnt <= w_cnt + 1'b1;
                end
            end
            if (adata_cmd) begin
                a_sh[a_cnt*32 +: 32] <= req_wdata;
                if (a_cnt == WB - 1) begin
                    a_cnt <= 0; a_pend <= 1;
                end else begin
                    a_cnt <= a_cnt + 1'b1;
                end
            end
            if (bdata_cmd) begin b_sh <= req_wdata; b_pend <= 1; end
            if (waddr_cmd) begin w_addr_cpu <= req_wdata[15:0]; w_cnt <= 0; end
            if (baddr_cmd) begin b_addr <= req_wdata[15:0]; end
            if (loaded_cmd) begin w_loaded_r <= 1; weights_ready <= 1'b1; end
            // 权重走 DMA 时，首组装载完成由 DMA 自己报告
            if (wt_en && dma_w_loaded) weights_ready <= 1'b1;

            // ---- DMA 配置（start 前写入生效）----
            if (write) begin
                case (rid)
                    R_DMA_AB:  dma_ab <= req_wdata;
                    R_DMA_AL:  dma_al <= req_wdata;
                    R_DMA_OB:  dma_ob <= req_wdata;
                    R_DMA_OL:  dma_ol <= req_wdata;
                    R_DMA_CFG: dma_en <= req_wdata[0];
                    R_WT_AB:  wt_ab  <= req_wdata;
                    R_WT_CFG: wt_en  <= req_wdata[0];
                    R_OUT_SEG:  out_seg_r  <= req_wdata;
                    R_OUT_ROWJ: out_rowj_r <= req_wdata;
                    R_OUT_ROWS: out_rows_r <= req_wdata[15:0];
                    default: ;
                endcase
            end

            // ---- 控制 ----
            if (clear_cmd) begin
                done <= 0; error <= 0; error_code <= 0; running <= 0;
                timer <= 0; cyc <= 0; w_addr_cpu <= 0; b_addr <= 0;
                w_cnt <= 0; a_cnt <= 0; o_cnt <= 0; o_valid <= 0;
                dma_finished <= 1'b0; eng_done_r <= 1'b0; weights_ready <= 1'b0;
            end
            if (start_cmd) begin
                cfg_pend <= 1; done <= 0; error <= 0; error_code <= 0;
                running <= 1; timer <= 0; cyc <= 0;
                dma_finished <= 1'b0; eng_done_r <= 1'b0; weights_ready <= 1'b0;
            end
            if (abort_cmd) begin
                running <= 0; error <= 1; error_code <= 2;                    // 2=中止
            end
            if (expired) begin
                running <= 0; error <= 1; error_code <= 4;                    // 4=超时
            end
            if (running || engine_busy) cyc <= cyc + 1'b1;
            if (running && !expired) timer <= timer + 1'b1;
            // DMA 出错立即上报；完成则以 dma_finished 为收尾条件之一
            if (dma_err != 0 && error_code == 0) begin
                error <= 1'b1; error_code <= dma_err;
            end
            // 引擎 done 是单拍脉冲，可能先于 DMA 收尾到达，必须分别锁存
            if (engine_done) eng_done_r <= 1'b1;
            if (dma_done)    dma_finished <= 1'b1;
            if (eng_done_r && (!dma_en || dma_finished)) begin
                done <= 1'b1; running <= 1'b0;
            end
        end
    end

    assign tpu_busy   = engine_busy;
    assign tpu_done   = done;
    assign tpu_pg_req = pg_load_req;
endmodule
