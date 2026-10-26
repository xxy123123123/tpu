`timescale 1ns/1ps
// stall 分类计数器验证
//
// 目的：上板后要靠这组计数器判断「算力 vs 带宽谁是天花板」，所以口径必须先
// 在仿真里钉死。本用例故意制造两种停顿：
//   [1] 激活欠喂：act_valid 每 3 拍才给 1 拍  -> 应计入 perf_starve / perf_act
//   [2] 输出背压：out_ready 每 5 拍拉低 1 拍  -> 应计入 perf_out
//   权重装载本身也要花周期                    -> 应计入 perf_wload
// 校验三点：
//   a) 完备性  total == comp + act + out + wload + drain（口径互斥且无遗漏）
//   b) 有效性  starve / out / wload / comp 全部 > 0（分类没退化成全 0）
//   c) 子关系  starve <= act + wload（starve 是正交标记，可落在任一 bucket）
module tb_tpu_perf;
    localparam P = 16, C = 8, WDEPTH = 128;
    localparam H = 4, W = 4, CIN = 8, COUT = 16;
    localparam NW = 9;                      // ncb(1) * nk(9)
    localparam NACT = H * W;                // ncb=1 -> 每遍 16 个 act beat
    localparam NEXP = H * W;                // npg=1 -> 16 个 out beat

    reg clk = 0, resetn = 0;
    always #5 clk = ~clk;

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

    wire [31:0] perf_total, perf_comp, perf_act, perf_starve,
                perf_out, perf_wload, perf_drain;

    tpu_conv_engine #(
        .P(P), .C(C), .MAX_W(256), .NCBMAX(8),
        .WDEPTH(WDEPTH), .MAX_COUT(256)
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
        .busy(busy), .done(done),
        .perf_total(perf_total), .perf_comp(perf_comp), .perf_act(perf_act),
        .perf_starve(perf_starve), .perf_out(perf_out),
        .perf_wload(perf_wload), .perf_drain(perf_drain)
    );

    integer ai, ei, wi, bi, errors, cyc;

    // ---------- 权重组预取（响应 pg_load_req）----------
    reg loading;
    always @(posedge clk) begin
        if (!resetn) begin
            w_valid <= 0; w_loaded <= 0; wi <= 0; loading <= 0;
        end else begin
            w_valid <= 0;
            if (pg_load_req && !loading && !w_loaded) begin
                loading <= 1; wi <= 0;
            end
            if (loading) begin
                if (wi < P * NW) begin
                    w_valid <= 1;
                    w_addr  <= (wi / NW) * WDEPTH + (wi % NW);
                    w_data  <= {C*8{1'b1}};                 // 权重全 1
                    wi      <= wi + 1;
                end else begin
                    loading  <= 0;
                    w_loaded <= 1;
                end
            end
            if (!pg_load_req && w_loaded) w_loaded <= 0;
        end
    end

    // ---------- 激活：故意每 3 拍才给 1 拍（制造欠喂）----------
    assign act_data  = {C*8{1'b1}};                          // 激活全 1
    assign act_valid = (ai < NACT) && (cyc % 3 == 0);
    always @(posedge clk) begin
        if (!resetn) ai <= 0;
        else if (act_valid && act_ready) ai <= ai + 1;
    end

    // ---------- 输出：故意每 5 拍拉低 1 拍（制造背压）----------
    always @(posedge clk) begin
        if (!resetn) begin ei <= 0; errors <= 0; end
        else if (out_valid && out_ready) ei <= ei + 1;
    end

    initial begin
        cfg_valid = 0; b_valid = 0; b_addr = 0; b_data = 0;
        cfg_stride = 2'd1; cfg_pad = 1'b1; cfg_relu = 1'b0;
        cfg_ksize = 1'b0; cfg_shift = 5'd0;
        cyc = 0;
        #100; resetn = 1; #20;

        // bias 全 0
        for (bi = 0; bi < COUT; bi = bi + 1) begin
            b_valid <= 1; b_addr <= bi[7:0]; b_data <= 32'd0;
            @(posedge clk);
        end
        b_valid <= 0;

        cfg_valid <= 1;
        cfg_h <= H; cfg_w <= W; cfg_cin <= CIN; cfg_cout <= COUT;
        @(posedge clk);
        cfg_valid <= 0;

        while (!done) begin
            @(posedge clk);
            cyc = cyc + 1;
            out_ready = (cyc % 5 != 0);
            if (cyc > 20000) begin
                $display("  [FAIL] 20000 拍未见 done");
                errors = errors + 1;
                $finish;
            end
        end
        @(posedge clk);

        $display("[1] 计数 total=%0d comp=%0d act=%0d starve=%0d out=%0d wload=%0d drain=%0d",
                 perf_total, perf_comp, perf_act, perf_starve,
                 perf_out, perf_wload, perf_drain);
        $display("    输出 beat 数 = %0d（期望 %0d）", ei, NEXP);
        if (ei !== NEXP) begin
            $display("  [FAIL] 输出 beat 数不符"); errors = errors + 1;
        end

        // a) 完备性
        if (perf_total !== perf_comp + perf_act + perf_out + perf_wload + perf_drain) begin
            $display("  [FAIL] 口径不完备: total=%0d 分项和=%0d",
                     perf_total, perf_comp + perf_act + perf_out + perf_wload + perf_drain);
            errors = errors + 1;
        end else begin
            $display("  [OK] 口径完备: total == comp+act+out+wload+drain");
        end

        // b) 有效性
        if (!(perf_starve > 0)) begin $display("  [FAIL] starve=0，欠喂没被计到"); errors = errors + 1; end
        if (!(perf_out    > 0)) begin $display("  [FAIL] out=0，背压没被计到");   errors = errors + 1; end
        if (!(perf_wload  > 0)) begin $display("  [FAIL] wload=0，等权重没计到"); errors = errors + 1; end
        if (!(perf_comp   > 0)) begin $display("  [FAIL] comp=0，计算没计到");     errors = errors + 1; end

        // c) 子关系：starve 是正交标记，可能落在 wload 里（装载权重时同时在填行缓冲）
        if (!(perf_starve <= perf_act + perf_wload)) begin
            $display("  [FAIL] starve(%0d) > act+wload(%0d)，子关系不成立",
                     perf_starve, perf_act + perf_wload);
            errors = errors + 1;
        end else begin
            $display("  [OK] 子关系成立: starve(%0d) <= act+wload(%0d)",
                     perf_starve, perf_act + perf_wload);
        end

        if (errors == 0) $display("\n==== tb_tpu_perf PASS ====");
        else             $display("\n==== tb_tpu_perf FAIL (%0d 处) ====", errors);
        $finish;
    end
endmodule
