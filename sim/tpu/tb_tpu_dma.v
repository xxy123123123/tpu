`timescale 1ns/1ps
// TPU DMA 端到端验证：CPU 只做配置，激活从 DDR 读、卷积结果写回 DDR。
//
// 场景（可手算）：
//   输入 4x4xC(8)，NHWC 布局，像素值全 1，全部放在 DDR 0x0001_0000
//   权重 3x3x8 全 1，bias 全 0，pad=1，stride=1，relu=0，shift=0
//   => 每个输出通道 = 窗口内有效 tap 数 x 8
//        角像素 4 taps -> 32 | 边像素 6 taps -> 48 | 内部像素 9 taps -> 72
//   输出 4x4xP(16) = 256 字节写回 DDR 0x0002_0000
// 期望：256 字节中 32 出现 64 次、48 出现 128 次、72 出现 64 次，其余为 0。
module tb_tpu_dma;
    localparam P = 16, C = 8;
    localparam H = 4, W = 4;
    localparam ACT_ADDR = 32'h0001_0000;
    localparam OUT_ADDR = 32'h0002_0000;
    localparam ACT_BYTES = H * W * C;          // 128
    localparam OUT_BYTES = H * W * P;          // 256

    reg clk = 0, resetn = 0;
    always #5 clk = ~clk;

    // ---------------- MMIO 主设备口 ----------------
    reg        req_valid = 0, req_write = 0;
    reg [1:0]  req_size = 2'd2;
    reg [31:0] req_addr = 0, req_wdata = 0;
    reg [3:0]  req_wstrb = 4'hf;
    wire       req_ready, rsp_valid, rsp_error;
    wire [31:0] rsp_rdata;

    // ---------------- DDR 直通口（接从设备模型）----------------
    wire [27:0]  araddr, awaddr;
    wire         arvalid, awvalid;
    wire [255:0] wdata;
    wire         arready, awready, wready;     // 由反压节拍器驱动
    reg          rvalid, rlast;
    reg  [3:0]   arid;
    reg  [255:0] rdata;
    reg          read_protocol_error;

    wire tpu_busy, tpu_done, tpu_pg_req;

    tpu_mmio #(.P(P), .C(C)) u_dut (
        .clk(clk), .resetn(resetn),
        .req_valid(req_valid), .req_write(req_write), .req_size(req_size),
        .req_addr(req_addr), .req_wdata(req_wdata), .req_wstrb(req_wstrb),
        .req_ready(req_ready), .rsp_valid(rsp_valid), .rsp_error(rsp_error),
        .rsp_rdata(rsp_rdata),
        .araddr(araddr), .awaddr(awaddr),
        .arvalid(arvalid), .awvalid(awvalid), .wdata(wdata),
        .arready(arready), .rvalid(rvalid), .rlast(rlast),
        .awready(awready), .wready(wready),
        .arid(arid), .rdata(rdata), .read_protocol_error(read_protocol_error),
        .wt_araddr(), .wt_arvalid(), .wt_arready(1'b0),
        .wt_rvalid(1'b0), .wt_rlast(1'b1), .wt_rid(4'd0),
        .wt_rdata(256'd0), .wt_read_protocol_error(1'b0),
        .tpu_busy(tpu_busy), .tpu_done(tpu_done), .tpu_pg_req(tpu_pg_req)
    );

    // ================= AXI 从设备模型（256 位 / 32 字节，带反压）=================
    reg [7:0] mem [0:262143];                 // 256 KB，字节寻址
    reg [31:0] rd_byte;
    reg        rd_pend;
    reg        wdiv, adiv;                    // 反压节拍器
    integer    k;

    task automatic rd_beat(input [31:0] ba, output [255:0] d);
        begin
            for (k = 0; k < 32; k = k + 1) d[k*8 +: 8] = mem[ba + k];
        end
    endtask

    initial begin
        rvalid = 0; rlast = 1; arid = 0; rdata = 0;
        read_protocol_error = 0;
        rd_pend = 0; wdiv = 0; adiv = 0;
    end

    // 读：AR 握上后隔一拍给数据；arready 隔拍有效以制造反压
    always @(posedge clk) begin
        if (!resetn) begin
            rd_pend <= 0; rvalid <= 0;
        end else begin
            rvalid <= 0;
            if (rd_pend) begin
                rd_pend <= 0;
                rd_beat(rd_byte, rdata);
                rvalid  <= 1'b1;
                rlast   <= 1'b1;
            end else if (arvalid && arready) begin
                rd_byte <= araddr << 2;
                rd_pend <= 1'b1;
            end
        end
    end
    // 写：AW 锁存地址后，W 数据可能在之后的拍才送达（awvalid 已拉低，wdata 仍有效）
    //     —— 与 Pango native 口 "no WVALID" 的语义一致，wdata 必须保持到 wready。
    reg        wr_armed;
    reg [31:0] wr_byte;
    always @(posedge clk) begin
        if (!resetn) begin
            wdiv <= 0; wr_armed <= 0; wr_byte <= 0;
        end else begin
            wdiv <= ~wdiv;
            if (awvalid && awready) begin
                wr_byte  <= awaddr << 2;
                wr_armed <= 1'b1;
            end
            if (wr_armed && wready) begin
                for (k = 0; k < 32; k = k + 1)
                    mem[wr_byte + k] <= wdata[k*8 +: 8];
                wr_armed <= 1'b0;
            end
        end
    end
    always @(posedge clk) begin
        if (!resetn) begin adiv <= 0; end
        else adiv <= ~adiv;
    end
    assign arready = adiv;      // 读地址通道隔拍可用
    assign awready = 1'b1;      // 写地址通道常开
    assign wready  = wdiv;      // 写数据通道隔拍可用

    // ================= CPU BFM =================
    task mmio_write(input [31:0] a, input [31:0] d);
        begin
            req_valid <= 1'b1; req_write <= 1'b1;
            req_addr <= a; req_wdata <= d;
            req_size <= 2'd2; req_wstrb <= 4'hf;
            @(posedge clk);
            while (!req_ready) @(posedge clk);
            req_valid <= 1'b0;
            @(posedge clk);
        end
    endtask

    task mmio_read(input [31:0] a, output [31:0] d);
        begin
            req_valid <= 1'b1; req_write <= 1'b0;
            req_addr <= a; req_wstrb <= 4'hf; req_size <= 2'd2;
            @(posedge clk);
            while (!req_ready) @(posedge clk);
            req_valid <= 1'b0;
            @(posedge clk);
            while (!rsp_valid) @(posedge clk);
            d = rsp_rdata;
            @(posedge clk);
        end
    endtask

    // 引擎的 w_pend / b_pend 未清时写数据会被拒（rsp_error=1 且数据丢弃），
    // 写之前必须先轮询 STATUS。
    reg [31:0] st;
    task wait_wfree();
        begin
            mmio_read(32'h08, st);
            while (st[4]) mmio_read(32'h08, st);     // bit4 = w_pend
        end
    endtask
    task wait_bfree();
        begin
            mmio_read(32'h08, st);
            while (st[3]) mmio_read(32'h08, st);     // bit3 = b_pend
        end
    endtask

    // ================= 主流程 =================
    integer i, p, k2, errors = 0;
    reg [31:0] rd;
    reg finished = 1'b0;
    integer cnt32, cnt48, cnt72, cntother;

    initial begin
        for (i = 0; i < OUT_BYTES; i = i + 1) mem[OUT_ADDR + i] = 0;
        // 激活：全 1
        for (i = 0; i < ACT_BYTES; i = i + 1) mem[ACT_ADDR + i] = 8'd1;

        repeat (6) @(posedge clk);
        resetn = 1'b1;
        repeat (6) @(posedge clk);

        mmio_read(32'h00, rd);
        $display("[1] ID = %08h %s", rd, (rd === 32'h54505531) ? "OK" : "FAIL");
        if (rd !== 32'h54505531) errors = errors + 1;

        // ---- 层配置 ----
        mmio_write(32'h0C, {16'd4, 16'd4});        // CFG_HW: h=4 w=4
        mmio_write(32'h10, {16'd8, 16'd16});       // CFG_CC: cin=8 cout=16
        mmio_write(32'h14, 32'h0000_0005);         // CFG_MD: stride=1 pad=1 k=3 relu=0 shift=0
        $display("[2] 层配置写入完成");

        // ---- DMA 配置 ----
        mmio_write(32'h40, ACT_ADDR);
        mmio_write(32'h44, ACT_BYTES);
        mmio_write(32'h48, OUT_ADDR);
        mmio_write(32'h4C, OUT_BYTES);
        mmio_write(32'h50, 32'd1);                 // dma_en = 1
        mmio_read(32'h54, rd);
        $display("[4] DMA_ST = %08h (bit0=dma_en)", rd);
        if (rd[0] !== 1'b1) begin $display("  [FAIL] dma_en 未生效"); errors = errors + 1; end

        // ---- start ----
        mmio_write(32'h04, 32'd1);
        $display("[5] start 已下发（权重/bias 在 start 后按背压写入）");

        // ---- bias：16 个通道，全 0（b_pend 背压，逐条等）----
        for (i = 0; i < 16; i = i + 1) begin
            wait_bfree();
            mmio_write(32'h24, i);                 // BADDR
            mmio_write(32'h20, 32'd0);             // BDATA
        end

        // ---- 主循环：响应权重预取，等 done ----
        i = 0; finished = 1'b0;
        while (!finished && i < 3000) begin
            mmio_read(32'h08, rd);                 // STATUS
            if (rd[7]) begin                       // pg_load_req
                // P 个 lane，每 lane 9 taps；每 beat C=8 字节 -> 2 个 32 位字
                // 引擎权重地址 = p*WDEPTH + idx，WDEPTH=512，必须按 lane 分段写
                for (p = 0; p < P; p = p + 1) begin
                    mmio_write(32'h1C, p * 512);   // WADDR
                    for (k2 = 0; k2 < 9; k2 = k2 + 1) begin
                        wait_wfree();
                        mmio_write(32'h18, 32'h01010101);
                        mmio_write(32'h18, 32'h01010101);
                    end
                end
                mmio_write(32'h30, 32'd1);         // LDCTL -> w_loaded 脉冲
                $display("    权重组已响应 pg_load_req（%0d lane x 9 tap）", P);
            end
            if (rd[1]) finished = 1'b1;            // done
            i = i + 1;
        end
        $display("    轮询 %0d 轮", i);
        mmio_read(32'h08, rd);
        $display("[6] STATUS = %08h (bit1=done bit0=busy bit6=error)", rd);
        if (rd[1] !== 1'b1) begin $display("  [FAIL] 未见到 done"); errors = errors + 1; end
        if (rd[6] !== 1'b0) begin $display("  [FAIL] error 置起"); errors = errors + 1; end
        mmio_read(32'h38, rd);
        $display("    ERR = %0d  CYC = (见下)", rd);
        mmio_read(32'h34, rd);
        $display("    CYC = %0d", rd);

        // ---- 校验写回 DDR 的结果 ----
        cnt32 = 0; cnt48 = 0; cnt72 = 0; cntother = 0;
        for (i = 0; i < OUT_BYTES; i = i + 1) begin
            case (mem[OUT_ADDR + i])
                8'd32: cnt32 = cnt32 + 1;
                8'd48: cnt48 = cnt48 + 1;
                8'd72: cnt72 = cnt72 + 1;
                default: cntother = cntother + 1;
            endcase
        end
        $display("[7] 输出统计: 32->%0d  48->%0d  72->%0d  其它->%0d",
                 cnt32, cnt48, cnt72, cntother);
        $display("    期望    : 32->64   48->128  72->64   其它->0");
        if (cnt32 !== 64 || cnt48 !== 128 || cnt72 !== 64 || cntother !== 0) begin
            $display("  [FAIL] 输出内容不符");
            errors = errors + 1;
            for (i = 0; i < 16; i = i + 1)
                $display("    px%0d  ch0..15 = %02h %02h %02h %02h %02h %02h %02h %02h",
                         i, mem[OUT_ADDR+i*16+0], mem[OUT_ADDR+i*16+1],
                         mem[OUT_ADDR+i*16+2], mem[OUT_ADDR+i*16+3],
                         mem[OUT_ADDR+i*16+4], mem[OUT_ADDR+i*16+5],
                         mem[OUT_ADDR+i*16+6], mem[OUT_ADDR+i*16+7]);
        end else begin
            $display("  [OK] 输出内容与手算一致");
        end

        if (errors == 0) $display("\n==== tb_tpu_dma PASS ====");
        else             $display("\n==== tb_tpu_dma FAIL (%0d 处) ====", errors);
        $finish;
    end
endmodule
