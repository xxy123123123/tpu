// TPU 布局布线评估顶层 v13：P=32, C=32（MAC 阵列相对 v10 的 P32C16 翻倍）
//
// 动机：v10 实测 APM 只用了 128/740(17.3%)、DRM 32/415(7.7%)，算力远未吃满。
//      v10 关键路径是 lmem(BRAM) -> APM 乘法器，Logic Levels=1，说明延迟
//      几乎全在硬核与时钟网络上，再切流水已无空间（v11 插寄存器导致 APM
//      打包退化 128->512）。于是转向「以面积换吞吐」：把 C 从 16 提到 32。
//
// 为什么是 C 翻倍而不是 P 翻倍：
//   - P=64 会让 out_data 变成 512 位，现有 tpu_dma 的 32 字节 AXI beat
//     （OUT_BPW = 32/P）不再成立，DMA 要大改；
//   - C=32 时 act_data 恰好 256 位 = 1 个 AXI beat，ACT_BPB = 32/32 = 1，
//     tpu_dma 无需改动即可兼容。
//
// 保真手段（防止综合器剪掉待测逻辑）：
//   [1] 所有输入由 LFSR 驱动（非常量），避免常量传播把 MAC 阵列/加法树简化掉；
//   [2] 所有输出做 XOR 归约引到 probe，避免无扇出的逻辑被优化删除。
// 注意：本顶层只用于时序/资源评估，不做功能仿真（功能由 sim/tpu 回归保证）。
module tpu_pnr_wrap_v13 (
    input  wire       clk,
    input  wire       resetn,
    output wire [7:0] probe
);
    // ---- LFSR：驱动全部输入，保证逻辑不被常量传播剪掉 ----
    reg [31:0] lfsr;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) lfsr <= 32'hACE1_2345;
        else         lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end

    // ---- 配置接口 ----
    wire        cfg_valid = lfsr[0];
    wire [15:0] cfg_h     = {2'b00, lfsr[13:0]};
    wire [15:0] cfg_w     = {2'b00, lfsr[27:14]};
    wire [15:0] cfg_cin   = {8'd0,  lfsr[7:0]};
    wire [15:0] cfg_cout  = {8'd0,  lfsr[15:8]};
    wire [1:0]  cfg_stride = lfsr[17:16];
    wire        cfg_pad   = lfsr[18];
    wire        cfg_ksize = lfsr[19];
    wire        cfg_relu  = lfsr[20];
    wire [4:0]  cfg_shift = lfsr[25:21];

    // ---- 权重 / bias / 激活流（v13: C=32 -> 256 bit）----
    wire        w_valid   = lfsr[26];
    wire [15:0] w_addr    = lfsr[15:0];
    wire [255:0] w_data   = {8{lfsr[31:0]}};          // C*8 = 256 bit
    wire        b_valid   = lfsr[28];
    wire [7:0]  b_addr    = lfsr[7:0];
    wire signed [31:0] b_data = lfsr[31:0];
    wire        act_valid = lfsr[29];
    wire [255:0] act_data = {8{lfsr[31:0]}};          // C*8 = 256 bit
    wire        out_ready = lfsr[30];
    wire        w_loaded  = lfsr[24];

    // ---- 引擎输出 ----
    wire        cfg_ready, w_ready, b_ready, act_ready;
    wire        out_valid;
    wire [255:0] out_data;                              // P*8 = 256 bit
    wire [15:0] load_pg, cur_pg;
    wire        load_bank, pg_load_req, busy, done;

    tpu_conv_engine #(
        .P(32), .C(32), .MAX_W(256), .NCBMAX(8),
        .WDEPTH(128), .MAX_COUT(256), .NBUF(5)
    ) u (
        .clk(clk), .resetn(resetn),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_h(cfg_h), .cfg_w(cfg_w), .cfg_cin(cfg_cin), .cfg_cout(cfg_cout),
        .cfg_stride(cfg_stride), .cfg_pad(cfg_pad), .cfg_relu(cfg_relu),
        .cfg_ksize(cfg_ksize), .cfg_shift(cfg_shift),
        .w_valid(w_valid), .w_ready(w_ready), .w_addr(w_addr), .w_data(w_data),
        .b_valid(b_valid), .b_ready(b_ready), .b_addr(b_addr), .b_data(b_data),
        .act_valid(act_valid), .act_ready(act_ready), .act_data(act_data),
        .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
        .w_loaded(w_loaded), .load_pg(load_pg), .load_bank(load_bank),
        .pg_load_req(pg_load_req), .cur_pg(cur_pg),
        .busy(busy), .done(done)
    );

    // ---- 归约输出：所有输出位都影响 probe，保证逻辑不被剪 ----
    assign probe[0] = ^out_data;
    assign probe[1] = ^load_pg ^ ^cur_pg;
    assign probe[2] = load_bank ^ pg_load_req ^ busy ^ done;
    assign probe[3] = cfg_ready ^ w_ready ^ b_ready ^ act_ready ^ out_valid;
    assign probe[7:4] = 4'b0;
endmodule
