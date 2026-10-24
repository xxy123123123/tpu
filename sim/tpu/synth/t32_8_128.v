// TPU 综合验证顶层：P32 x C16 = 512 MAC/cycle（PG2L200H 甜点配置）
// 仅用于独立综合评估资源与时序，不参与仿真。
module t32_8_128 (
    input  wire clk,
    input  wire resetn,
    input  wire        cfg_valid,
    output wire        cfg_ready,
    input  wire [15:0] cfg_h,
    input  wire [15:0] cfg_w,
    input  wire [15:0] cfg_cin,
    input  wire [15:0] cfg_cout,
    input  wire [1:0]  cfg_stride,
    input  wire        cfg_pad,
    input  wire        cfg_ksize,
    input  wire        cfg_relu,
    input  wire [4:0]  cfg_shift,
    input  wire            w_valid,
    output wire            w_ready,
    input  wire [15:0]     w_addr,
    input  wire [8*8-1:0] w_data,
    input  wire              b_valid,
    output wire              b_ready,
    input  wire [7:0]        b_addr,
    input  wire signed [31:0] b_data,
    input  wire              act_valid,
    output wire              act_ready,
    input  wire [8*8-1:0]   act_data,
    output wire              out_valid,
    input  wire              out_ready,
    output wire [32*8-1:0]   out_data,
    input  wire        w_loaded,
    output wire [15:0] load_pg,
    output wire        load_bank,
    output wire        pg_load_req,
    output wire [15:0] cur_pg,
    output wire busy,
    output wire done
);
    tpu_conv_engine #(
        .P(32), .C(8), .MAX_W(256), .NCBMAX(8),
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
endmodule
