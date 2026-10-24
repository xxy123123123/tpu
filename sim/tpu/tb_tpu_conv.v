`timescale 1ns/1ps
`include "params.vh"

module tb_tpu_conv;
    localparam P        = TB_P;
    localparam C        = TB_C;
    localparam MAX_W    = TB_MAX_W;
    localparam NCBMAX   = TB_NCBMAX;
    localparam WDEPTH   = TB_WDEPTH;
    localparam MAX_COUT = TB_MAX_COUT;
    localparam NACT     = TB_NACT;
    localparam NEXP     = TB_NEXP;
    localparam NPG      = TB_NPG;
    localparam NCB      = TB_NCB;

    reg clk = 0, resetn = 0;
    always #5 clk = ~clk;

    reg [C*8-1:0] act_mem [0:NACT-1];
    reg [C*8-1:0] wgt_mem [0:NPG*P*TB_NW-1];
    reg [31:0]    bias_mem [0:TB_COUT-1];
    reg [P*8-1:0] exp_mem [0:NEXP-1];

    initial begin
        $readmemh("act.hex",  act_mem);
        $readmemh("wgt.hex",  wgt_mem);
        $readmemh("bias.hex", bias_mem);
        $readmemh("exp.hex",  exp_mem);
    end

    reg             cfg_valid;
    wire            cfg_ready;
    reg  [15:0]     cfg_h, cfg_w, cfg_cin, cfg_cout;
    reg  [1:0]      cfg_stride;
    reg             cfg_pad, cfg_relu, cfg_ksize;
    reg  [4:0]      cfg_shift;

    reg             w_valid;
    wire            w_ready;
    reg  [15:0]     w_addr;
    reg  [C*8-1:0]  w_data;

    reg             b_valid;
    wire            b_ready;
    reg  [7:0]      b_addr;
    reg  [31:0]     b_data;

    wire            act_valid;
    wire            act_ready;
    wire [C*8-1:0]  act_data;

    wire            out_valid;
    reg             out_ready;
    wire [P*8-1:0]  out_data;

    reg             w_loaded;
    wire [15:0]     load_pg;
    wire            load_bank;
    wire            pg_load_req;
    wire [15:0]     cur_pg;
    wire            busy, done;

    tpu_conv_engine #(
        .P(P), .C(C), .MAX_W(MAX_W), .NCBMAX(NCBMAX),
        .WDEPTH(WDEPTH), .MAX_COUT(MAX_COUT)
    ) u (
        .clk(clk), .resetn(resetn),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_h(cfg_h), .cfg_w(cfg_w), .cfg_cin(cfg_cin), .cfg_cout(cfg_cout),
        .cfg_stride(cfg_stride), .cfg_pad(cfg_pad), .cfg_relu(cfg_relu),
        .cfg_ksize(cfg_ksize),
        .cfg_shift(cfg_shift),
        .w_valid(w_valid), .w_ready(w_ready), .w_addr(w_addr), .w_data(w_data),
        .b_valid(b_valid), .b_ready(b_ready), .b_addr(b_addr), .b_data(b_data),
        .act_valid(act_valid), .act_ready(act_ready), .act_data(act_data),
        .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
        .w_loaded(w_loaded), .load_pg(load_pg), .load_bank(load_bank),
        .pg_load_req(pg_load_req), .cur_pg(cur_pg),
        .busy(busy), .done(done)
    );

    integer ai, ei, wi, bi, errors;
    integer cyc_all, cyc_comp, cyc_wload, cyc_drain, cyc_stall;

    always @(posedge clk) begin
        if (!resetn) begin
            cyc_all <= 0; cyc_comp <= 0; cyc_wload <= 0; cyc_drain <= 0; cyc_stall <= 0;
        end else if (busy) begin
            cyc_all <= cyc_all + 1;
            case (u.state)
                3'd2: begin                                     // S_COMP
                    cyc_comp <= cyc_comp + 1;
                    if (!u.row_ready) cyc_stall <= cyc_stall + 1;   // 行未就绪停拍
                end
                3'd1: cyc_wload <= cyc_wload + 1;   // S_WLOAD（权重未跟上的停顿）
                3'd3: cyc_drain <= cyc_drain + 1;   // S_DRAIN
            endcase
        end
    end

    // ---------- 权重组预取：响应 pg_load_req，写到 load_bank ----------
    reg loading;
    reg [15:0] pg_latch;
    always @(posedge clk) begin
        if (!resetn) begin
            w_valid <= 0; w_loaded <= 0; wi <= 0; loading <= 0; pg_latch <= 0;
        end else begin
            w_valid <= 0;
            if (pg_load_req && !loading && !w_loaded) begin
                loading <= 1; wi <= 0; pg_latch <= load_pg;
            end
            if (loading) begin
                if (wi < P * TB_NW) begin
                    w_valid <= 1;
                    // RTL 侧仍按 WDEPTH 寻址: w_addr = p*WDEPTH + idx
                    w_addr  <= (wi / TB_NW) * WDEPTH + (wi % TB_NW);
                    w_data  <= wgt_mem[pg_latch * (P * TB_NW) + wi];
                    wi      <= wi + 1;
                end else begin
                    loading  <= 0;
                    w_loaded <= 1;
                end
            end
            if (!pg_load_req && w_loaded) w_loaded <= 0;
        end
    end

    // 连续赋值驱动，避免非阻塞赋值造成的数据比握手慢一拍（会整体错位）
    assign act_data  = (ai < NACT) ? act_mem[ai] : {C * 8{1'b0}};
    assign act_valid = (ai < NACT);
    always @(posedge clk) begin
        if (!resetn) ai <= 0;
        else if (act_valid && act_ready) ai <= ai + 1;
    end

    // ---------- 输出收集与比对 ----------
    always @(posedge clk) begin
        if (!resetn) begin
            ei <= 0; errors <= 0; out_ready <= 1;
        end else if (out_valid && out_ready) begin
            if (ei < NEXP) begin
                if (out_data !== exp_mem[ei]) begin
                    errors <= errors + 1;
                    if (errors <= 5)
                        $display("MISMATCH beat=%0d got=%h exp=%h", ei, out_data, exp_mem[ei]);
                end
                ei <= ei + 1;
            end
        end
    end

    initial begin
        resetn = 0; cfg_valid = 0; w_valid = 0; b_valid = 0;
        out_ready = 1; w_loaded = 0; loading = 0;
        cfg_h = 0; cfg_w = 0; cfg_cin = 0; cfg_cout = 0;
        cfg_stride = 0; cfg_pad = 0; cfg_relu = 0; cfg_shift = 0; cfg_ksize = 3;
        w_addr = 0; w_data = 0; b_addr = 0; b_data = 0;
        #100;
        resetn = 1;
        #20;

        // bias
        for (bi = 0; bi < TB_COUT; bi = bi + 1) begin
            b_valid <= 1; b_addr <= bi; b_data <= bias_mem[bi];
            @(posedge clk);
        end
        b_valid <= 0;

        // 层配置
        cfg_valid <= 1;
        cfg_h     <= TB_H;   cfg_w    <= TB_W;
        cfg_cin   <= TB_CIN; cfg_cout <= TB_COUT;
        cfg_stride<= TB_STRIDE; cfg_pad <= TB_PAD[0];
        cfg_relu  <= TB_RELU[0]; cfg_shift <= TB_SHIFT;
        cfg_ksize <= (TB_KSIZE == 1) ? 1'b1 : 1'b0;
        @(posedge clk);
        cfg_valid <= 0;

        wait (done);
        #200;
        if (ei != NEXP)
            $display("RESULT: FAIL 输出 beat 数不符 got=%0d exp=%0d", ei, NEXP);
        else if (ai != NACT)
            $display("RESULT: FAIL 输入 beat 数不符 got=%0d exp=%0d", ai, NACT);
        else if (errors != 0)
            $display("RESULT: FAIL (%0d / %0d 字节不匹配)", errors, NEXP * P);
        else begin
            begin : perf
                integer macs;
                macs = TB_HOUT * TB_WOUT * TB_CIN * TB_COUT * TB_KSIZE * TB_KSIZE;
                $display("PERF macs=%0d comp=%0d stall=%0d wload=%0d drain=%0d total=%0d | MAC/cyc(comp)=%.2f 理论=%0d | act_beats=%0d 一遍=%0d v1需%0d 省 %.0f%%", macs, cyc_comp, cyc_stall, cyc_wload, cyc_drain, cyc_all, (cyc_comp > 0) ? (macs * 1.0 / cyc_comp) : 0.0, P * C, ai, TB_NACT_TH, TB_NACT_V1, (TB_NACT_V1 > 0) ? (100.0 * (TB_NACT_V1 - ai) / TB_NACT_V1) : 0.0);
            end
            $display("RESULT: PASS P=%0d C=%0d K%0d %0dx%0dx%0d->%0d beats=%0d bytes=%0d",
                     P, C, TB_KSIZE, TB_H, TB_W, TB_CIN, TB_COUT, ei, ei * P);
        end
        $finish;
    end

`ifdef TB_TRACE
    initial begin
        #260;
        repeat (400) begin
            @(posedge clk);
            $display("st=%0d pg=%0d oy=%0d cb=%0d ox=%0d k=%0d vr=%0d fx=%0d bv=%b | actR=%b aV=%b req=%b lp=%0d lb=%0d wl=%0d outV=%b rd=%0d", u.state, u.pg, u.oy, u.cb, u.ox, u.k, u.vr, u.fx, u.bvalid, act_ready, act_valid, pg_load_req, load_pg, load_bank, w_loaded, out_valid, u.row_ready);
        end
    end
`endif

    initial begin
        #30000000;
        $display("RESULT: FAIL TIMEOUT ai=%0d/%0d ei=%0d/%0d", ai, NACT, ei, NEXP);
        $finish;
    end

endmodule
