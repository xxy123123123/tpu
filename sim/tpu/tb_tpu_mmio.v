`timescale 1ns/1ps
// tpu_mmio 端到端验证：模拟 CPU 用 MMIO 配置 TPU、灌权重/bias、喂激活、读回结果。
//
// 场景（刻意构造出可手算的期望值）：
//   P=4 C=4，3x3 卷积，pad=0，stride=1，4x4 输入 -> 2x2 输出
//   权重全 1、激活全 1、bias 全 0、shift=0、relu=0
//   => 每个输出通道 = C * 9 = 4 * 9 = 36 = 8'h24（窗口内全是有效 tap）
//   => 每个输出 beat（P=4 字节）= 32'h24242424
//
// 用法：
//   iverilog -g2012 -I myriscv -s tb_tpu_mmio -o t.vvp sim/tpu/tb_tpu_mmio.v myriscv/tpu_mmio.v myriscv/tpu_conv_engine.v
//   vvp t.vvp
module tb_tpu_mmio;
    localparam P = 4, C = 4, WDEPTH = 16, MAX_COUT = 16, NCBMAX = 2, MAX_W = 64, NBUF = 5;
    // 用例规模
    localparam H = 4, W = 4, CIN = 4, COUT = 4;
    localparam OH = H - 2, OW = W - 2;            // pad=0, k=3 -> 2x2
    localparam NACT = H * W;                      // 激活 beats（ncb = cin/C = 1）
    localparam NOUT = OH * OW;                    // 输出 beats（npg = cout/P = 1）
    localparam NWB  = 9;                          // 3x3 taps

    // 寄存器字节地址
    localparam A_ID = 8'h00, A_CTRL = 8'h04, A_STATUS = 8'h08, A_CFG_HW = 8'h0C,
               A_CFG_CC = 8'h10, A_CFG_MD = 8'h14, A_WDATA = 8'h18, A_WADDR = 8'h1C,
               A_BDATA = 8'h20, A_BADDR = 8'h24, A_ADATA = 8'h28, A_ODATA = 8'h2C,
               A_LDCTL = 8'h30, A_CYC = 8'h34, A_ERR = 8'h38, A_PARAM = 8'h3C;

    reg clk = 0, resetn = 0;
    reg        req_valid = 0, req_write = 0;
    reg [1:0]  req_size = 2'd2;
    reg [31:0] req_addr = 0, req_wdata = 0;
    reg [3:0]  req_wstrb = 4'hf;
    wire       req_ready;
    wire       rsp_valid, rsp_error;
    wire [31:0] rsp_rdata;
    wire       tpu_busy, tpu_done, tpu_pg_req;

    tpu_mmio #(
        .P(P), .C(C), .MAX_W(MAX_W), .NCBMAX(NCBMAX),
        .WDEPTH(WDEPTH), .MAX_COUT(MAX_COUT), .NBUF(NBUF)
    ) u_dut (
        .clk(clk), .resetn(resetn),
        .req_valid(req_valid), .req_write(req_write), .req_size(req_size),
        .req_addr(req_addr), .req_wdata(req_wdata), .req_wstrb(req_wstrb),
        .req_ready(req_ready), .rsp_valid(rsp_valid), .rsp_error(rsp_error),
        .rsp_rdata(rsp_rdata),
        .tpu_busy(tpu_busy), .tpu_done(tpu_done), .tpu_pg_req(tpu_pg_req)
    );

    always #5 clk = ~clk;

    integer errors = 0;
    integer i, j, p, k, n;
    reg [31:0] rd;
    reg [31:0] outbuf [0:15];

    // ---------------- MMIO 事务 ----------------
    task mmio_write(input [31:0] addr, input [31:0] data);
        reg done;
        begin
            req_valid <= 1'b1; req_write <= 1'b1;
            req_addr <= addr; req_wdata <= data;
            req_size <= 2'd2; req_wstrb <= 4'hf;
            done = 1'b0;
            while (!done) begin
                @(posedge clk);
                if (req_ready) begin done = 1'b1; req_valid <= 1'b0; end
            end
            while (!rsp_valid) @(posedge clk);
            if (rsp_error) begin
                $display("  [FAIL] MMIO write addr=%02h err", addr);
                errors = errors + 1;
            end
            @(posedge clk);
        end
    endtask

    task mmio_read(input [31:0] addr, output [31:0] data);
        reg done;
        begin
            req_valid <= 1'b1; req_write <= 1'b0;
            req_addr <= addr; req_wdata <= 32'b0;
            req_size <= 2'd2; req_wstrb <= 4'hf;
            done = 1'b0;
            while (!done) begin
                @(posedge clk);
                if (req_ready) begin done = 1'b1; req_valid <= 1'b0; end
            end
            while (!rsp_valid) @(posedge clk);
            if (rsp_error) begin
                $display("  [FAIL] MMIO read addr=%02h err", addr);
                errors = errors + 1;
            end
            data = rsp_rdata;
            @(posedge clk);
        end
    endtask

    initial begin
        // ---- 复位 ----
        repeat (4) @(posedge clk);
        resetn <= 1'b1;
        repeat (4) @(posedge clk);

        // ---- 1. 读 ID ----
        mmio_read(A_ID, rd);
        $display("[1] ID = %08h (期望 54505531)", rd);
        if (rd !== 32'h54505531) begin
            $display("  [FAIL] ID 不符"); errors = errors + 1;
        end

        // ---- 2. 层配置 ----
        mmio_write(A_CFG_HW, {H[15:0], W[15:0]});
        mmio_write(A_CFG_CC, {CIN[15:0], COUT[15:0]});
        // bit[1:0]=stride=1, bit[2]=pad=0, bit[3]=ksize=0(3x3), bit[4]=relu=0, bit[12:8]=shift=0
        mmio_write(A_CFG_MD, 32'h0000_0001);
        $display("[2] 层配置写入完成");

        // ---- 3. bias 全 0 ----
        for (i = 0; i < COUT; i = i + 1) begin
            mmio_write(A_BADDR, i);
            mmio_write(A_BDATA, 32'h0000_0000);
        end
        $display("[3] bias 写入完成 (%0d 项)", COUT);

        // ---- 4. 启动 ----
        mmio_write(A_CTRL, 32'h0000_0001);
        $display("[4] start 已下发");

        // ---- 5. 主循环：响应权重预取 / 收输出 / 喂激活 ----
        // 引擎的 w_addr = p*WDEPTH + idx，必须按 lane 分段写，不能一路自增。
        // 输出必须边算边收：o_valid 时若不读走，out_ready 会拉低把引擎背压住。
        j = 0;   // 已喂激活 beats
        n = 0;   // 已收输出 beats
        i = 0;
        while ((!tpu_done || n < NOUT) && i < 4000) begin
            mmio_read(A_STATUS, rd);
            if (rd[7]) begin                       // pg_load_req
                for (p = 0; p < P; p = p + 1) begin
                    mmio_write(A_WADDR, p * WDEPTH);
                    for (k = 0; k < NWB; k = k + 1)
                        mmio_write(A_WDATA, 32'h01010101);
                end
                mmio_write(A_LDCTL, 32'h0000_0001);  // w_loaded 脉冲
                $display("[5] 权重组已灌入（%0d lane x %0d tap）并置 w_loaded", P, NWB);
            end else if (rd[6] && n < NOUT) begin   // o_valid：有输出就收
                mmio_read(A_ODATA, rd);
                outbuf[n] = rd;
                n = n + 1;
            end else if (j < NACT) begin
                mmio_write(A_ADATA, 32'h01010101);
                j = j + 1;
            end
            i = i + 1;
        end
        $display("[5] 激活已喂 %0d/%0d beats，已收输出 %0d/%0d，done=%0b",
                 j, NACT, n, NOUT, tpu_done);

        // ---- 6. 收尾状态 ----
        mmio_read(A_STATUS, rd);
        $display("[6] STATUS=%08h done=%0b busy=%0b", rd, rd[1], rd[0]);
        if (!rd[1]) begin
            $display("  [FAIL] 未完成（done=0）"); errors = errors + 1;
        end
        mmio_read(A_ERR, rd);
        if (rd !== 0) begin
            $display("  [FAIL] 错误码 = %0d", rd); errors = errors + 1;
        end

        // ---- 7. 比对输出 ----
        $display("[7] 输出比对（%0d 个 beat）：", NOUT);
        for (i = 0; i < NOUT; i = i + 1) begin
            $display("    beat%0d = %08h (期望 24242424)", i, outbuf[i]);
            if (outbuf[i] !== 32'h24242424) begin
                $display("      [FAIL] 输出不符"); errors = errors + 1;
            end
        end
        if (n !== NOUT) begin
            $display("  [FAIL] 输出 beat 数 %0d != 期望 %0d", n, NOUT);
            errors = errors + 1;
        end

        mmio_read(A_CYC, rd);
        $display("[8] 计算周期数 = %0d", rd);

        if (errors == 0)
            $display("\n==== tb_tpu_mmio PASS ====");
        else
            $display("\n==== tb_tpu_mmio FAIL (%0d 处) ====", errors);
        $finish;
    end

    // 兜底：防止仿真挂死
    initial begin
        #2_000_000;
        $display("[TIMEOUT] 仿真超时");
        $finish;
    end
endmodule
