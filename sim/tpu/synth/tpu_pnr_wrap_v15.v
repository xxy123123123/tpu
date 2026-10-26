// TPU 布局布线评估顶层 v15：P=32, C=16（对照组）
//
// 与 v10/v13 的关键区别：**激励每个字节都不同**。
//   v10/v13 用 act_data = {8{lfsr}}（同一个 32 位字复制 8 份），C=32 时 32 个
//   激活字节两两相同；结果 v13(P32C32) 与 v10(P32C16) 的 LUT 只差 9 个、
//   APM 都是 128 —— 推断出的乘法器数倒是 512 -> 1024 翻倍了，但最终映射
//   规模没变，怀疑综合器把等价乘法归并了。
// 本版改用两个独立的 256 位字节移位寄存器：每周期整体左移 8 位并注入由
// 计数器混淆的新字节，保证 32 个字节互不相同、且随时间变化。
//
// 保真手段：
//   [1] 输入非常量，避免常量传播把 MAC 阵列剪掉；
//   [2] 输入字节两两不同，避免综合器归并等价乘法器；
//   [3] 输出逐位可观测，避免无扇出逻辑被删除。
// 只用于时序/资源评估，不做功能仿真（功能由 sim/tpu 回归保证）。
module tpu_pnr_wrap_v15 (
    input  wire       clk,
    input  wire       resetn,
    output wire [7:0] probe
);
    localparam P8 = 32 * 8;    // P*8 = 256
    localparam C8 = 16 * 8;    // C*8 = 128

    // ---- 字节移位寄存器：每周期左移 8 位并注入混淆字节 ----
    reg [255:0] ash, wsh;
    reg [7:0]   cnt;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            ash <= 256'h0b2a4968778ca0b9d2e70c2543e62a1f8d0c37b64e91a5d82f0b3e47596a1c2;
            wsh <= 256'h07385f8ca611d942f77ba0c512f8d6ab390c47e2519f3a0c58b6d1e27394f50;
            cnt <= 8'h00;
        end else begin
            cnt <= cnt + 8'd1;
            ash <= {ash[247:0], ash[255:248] ^ cnt ^ 8'h5a};
            wsh <= {wsh[247:0], wsh[255:248] ^ ~cnt ^ 8'h3c};
        end
    end

    // ---- 配置接口 ----
    wire        cfg_valid  = ash[0];
    wire [15:0] cfg_h      = {2'b00, ash[13:0]};
    wire [15:0] cfg_w      = {2'b00, ash[27:14]};
    wire [15:0] cfg_cin    = {8'd0,  ash[7:0]};
    wire [15:0] cfg_cout   = {8'd0,  ash[15:8]};
    wire [1:0]  cfg_stride = ash[17:16];
    wire        cfg_pad    = ash[18];
    wire        cfg_ksize  = ash[19];
    wire        cfg_relu   = ash[20];
    wire [4:0]  cfg_shift  = ash[25:21];

    // ---- 权重 / bias / 激活流 ----
    wire              w_valid   = ash[26];
    wire [15:0]       w_addr    = ash[15:0];
    wire [C8-1:0]     w_data    = wsh[C8-1:0];
    wire              b_valid   = ash[28];
    wire [7:0]        b_addr    = ash[7:0];
    wire signed [31:0] b_data   = ash[31:0];
    wire              act_valid = ash[29];
    wire [C8-1:0]     act_data  = ash[C8-1:0];
    wire              out_ready = ash[30];
    wire              w_loaded  = ash[24];

    // ---- 引擎输出 ----
    wire             cfg_ready, w_ready, b_ready, act_ready;
    wire             out_valid;
    wire [P8-1:0]    out_data;
    wire [15:0]      load_pg, cur_pg;
    wire             load_bank, pg_load_req, busy, done;

    tpu_conv_engine #(
        .P(32), .C(16), .MAX_W(256), .NCBMAX(8),
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

    // ---- 逐位可观测：256 位输出分 8 组各 32 位 XOR -> probe[7:0] ----
    wire [P8-1:0] mon = out_data ^ {P8{act_ready}};
    wire [7:0]    probe_raw;
    genvar gi;
    for (gi = 0; gi < 8; gi = gi + 1) begin : g_probe
        assign probe_raw[gi] = ^mon[gi*32 +: 32];
    end
    // 状态信号也要可观测，否则控制逻辑会被剪掉
    wire [7:0] st = {cfg_ready, w_ready, b_ready, act_ready,
                     out_valid, busy, done, pg_load_req};
    assign probe = probe_raw ^ {8{^st ^ ^load_pg ^ ^cur_pg ^ load_bank}};
endmodule
