`timescale 1ns/1ps
// TPU 权重 DMA 端到端验证：CPU 只写配置 + bias + start，权重由 DMA 从 DDR 搬。
//
// 与 tb_tpu_dma 的区别：后者权重靠 CPU 逐 beat 经 R_WDATA 灌（2300 次 MMIO），
// 本用例权重走 DDR，CPU 一次 R_WDATA 都不写，只在最后校验输出。
//
// 场景（可手算）：
//   输入 4x4x8（cin=8）全 1，放 DDR 0x0001_0000
//   cout=32 -> npg=2（两组，顺便验证乒乓与组地址递增）
//   权重布局 [cout][cb][k]，每单元 C=8 字节：
//       co 为偶数 -> 值 1；co 为奇数 -> 值 2
//     每组字节数 = P*ncb*nk*C = 16*1*9*8 = 1152（32 的倍数，天然对齐）
//   bias 全 0，pad=1 stride=1 relu=0 shift=0
//   => 输出通道 co、像素类型 t 的值 = 有效 tap 数 x 8 x w(co)
//        角 4 taps: 32*w | 边 6 taps: 48*w | 内部 9 taps: 72*w
//  4x4 输出中角 4 个、边 8 个、内 4 个，co 里偶奇各 16 个，故期望：
//        32 -> 4*16=64   64 -> 64    48 -> 8*16=128  96 -> 128
//        72 -> 64        144 -> 64
//  若 lane->cout 映射错位（所有 lane 拿到同一通道的权重），偶奇差异会消失，
//  统计立刻对不上——这是本用例的主要检测点。
module tb_tpu_wdma;
    localparam P = 16, C = 8;
    localparam H = 4, W = 4, CIN = 8, COUT = 32;
    localparam ACT_ADDR = 32'h0001_0000;
    localparam OUT_ADDR = 32'h0002_0000;
    localparam WT_ADDR  = 32'h0003_0000;
    localparam ACT_BYTES = H * W * CIN;         // 128
    localparam OUT_BYTES = H * W * COUT;        // 512
    localparam WT_BYTES  = COUT * CIN * 9;      // 2304（每组 1152）

    reg clk = 0, resetn = 0;
    always #5 clk = ~clk;

    // ---------------- MMIO 主设备口 ----------------
    reg        req_valid = 0, req_write = 0;
    reg [1:0]  req_size = 2'd2;
    reg [31:0] req_addr = 0, req_wdata = 0;
    reg [3:0]  req_wstrb = 4'hf;
    wire       req_ready, rsp_valid, rsp_error;
    wire [31:0] rsp_rdata;

    // ---------------- DDR 直通口 ----------------
    wire [27:0]  araddr, awaddr;
    wire         arvalid, awvalid;
    wire [255:0] wdata;
    wire         arready, awready, wready;
    // rvalid/rlast/rid/rdata 由读仲裁器分发（仲裁器只分发 valid，数据总线共享）
    wire         rvalid;
    wire [3:0]   arid;
    reg          read_protocol_error;
    // ---- 权重专用读口（与激活口并行）----
    wire [27:0]  wt_araddr;
    wire         wt_arvalid, wt_arready;
    wire         wt_rvalid;
    wire [3:0]   wt_rid;
    wire         wt_read_protocol_error;
    // 两条读口经仲裁器汇聚到同一个 DDR 从设备（与 soc_ddr3_top 完全同构）
    wire [27:0]  slv_araddr;
    wire [3:0]   slv_arid;
    wire         slv_arvalid, slv_arready;
    reg          slv_rvalid;
    reg  [255:0] slv_rdata;
    reg  [3:0]   slv_rid;
    // arlen 恒 0 => 单 beat => rlast 恒 1，直接常量即可
    wire         rlast      = 1'b1;
    wire         wt_rlast   = 1'b1;
    wire [255:0] rdata      = slv_rdata;
    wire [255:0] wt_rdata   = slv_rdata;
    assign arid    = slv_rid;
    assign wt_rid  = slv_rid;

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
        .wt_araddr(wt_araddr), .wt_arvalid(wt_arvalid), .wt_arready(wt_arready),
        .wt_rvalid(wt_rvalid), .wt_rlast(wt_rlast), .wt_rid(wt_rid),
        .wt_rdata(wt_rdata), .wt_read_protocol_error(wt_read_protocol_error),
        .tpu_busy(tpu_busy), .tpu_done(tpu_done), .tpu_pg_req(tpu_pg_req)
    );

    // ================= AXI 从设备模型（256 位 / 32 字节，带反压）=================
    reg [7:0] mem [0:262143];
    reg [31:0] rd_byte;
    reg        rd_pend;
    reg        wdiv, adiv;
    integer    k;

    task automatic rd_beat(input [31:0] ba, output [255:0] d);
        begin
            for (k = 0; k < 32; k = k + 1) d[k*8 +: 8] = mem[ba + k];
        end
    endtask

    initial begin
        read_protocol_error = 0;
        rd_pend = 0; wdiv = 0; adiv = 0; slv_rdata = 0; slv_rid = 0;
    end

    // 两条读口经**真实仲裁器**汇聚到同一个 DDR 从设备。
    // 为什么不用手写的 grant 组合逻辑：手写的版本踩过相位死锁 —— 反压信号
    // adiv 与轮转标记 last_owner 都是周期 2，相位一旦对齐就永远只授权同一
    // 个口（实测权重只读 4 个 beat 后彻底卡死）。真实仲裁器以「一次完整
    // AR→R」为粒度轮转，不存在这个问题，而且与 soc_ddr3_top 里的结构一致。
    preprocess_ddr_read_arbiter #(.PP_ID(4'd6)) u_wt_read_arbiter (
        .clk(clk), .resetn(resetn),
        .cpu_addr(araddr), .cpu_len(4'd0), .cpu_id(4'd5), .cpu_ap(1'b0),
        .cpu_valid(arvalid), .cpu_ready(arready), .cpu_rvalid(rvalid),
        .pp_addr(wt_araddr), .pp_valid(wt_arvalid), .pp_ready(wt_arready), .pp_rvalid(wt_rvalid),
        .araddr(slv_araddr), .arlen(), .arid(slv_arid), .arap(),
        .arvalid(slv_arvalid), .arready(slv_arready),
        .rvalid(slv_rvalid), .rlast(1'b1), .rid(slv_rid),
        .protocol_error(wt_read_protocol_error)
    );

    reg [3:0] rid_hold;
    always @(posedge clk) begin
        if (!resetn) begin
            rd_pend <= 0; slv_rvalid <= 0; rid_hold <= 0;
        end else begin
            slv_rvalid <= 0;
            if (rd_pend) begin
                rd_pend     <= 0;
                rd_beat(rd_byte, slv_rdata);
                slv_rvalid  <= 1'b1;
                slv_rid     <= rid_hold;
            end else if (slv_arvalid && slv_arready) begin
                rd_byte  <= slv_araddr << 2;
                rid_hold <= slv_arid;
                // rid 必须与 rvalid 同拍可见：仲裁器在返回拍就比对 rid，
                // 晚一拍会被判成协议错误并**永久锁死**整个读通道。
                slv_rid  <= slv_arid;
                rd_pend  <= 1'b1;
            end
        end
    end

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
    assign slv_arready = adiv;
    assign awready = 1'b1;
    assign wready  = wdiv;

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

    reg [31:0] st;
    task wait_bfree();
        begin
            mmio_read(32'h08, st);
            while (st[3]) mmio_read(32'h08, st);     // bit3 = b_pend
        end
    endtask

    // ================= 主流程 =================
    integer i, co, kk, errors = 0;
    reg [31:0] rd;
    reg finished = 1'b0;
    integer c32, c64, c48, c96, c72, c144, cother;

    initial begin
        // ---- 内存初始化 ----
        for (i = 0; i < OUT_BYTES; i = i + 1) mem[OUT_ADDR + i] = 0;
        for (i = 0; i < ACT_BYTES; i = i + 1) mem[ACT_ADDR + i] = 8'd1;
        // 权重 [cout][cb][k]，每单元 C 字节；co 偶 -> 1，co 奇 -> 2
        for (co = 0; co < COUT; co = co + 1)
            for (kk = 0; kk < 9; kk = kk + 1)
                for (i = 0; i < C; i = i + 1)
                    mem[WT_ADDR + co*72 + kk*C + i] = (co[0] == 1'b0) ? 8'd1 : 8'd2;

        repeat (6) @(posedge clk);
        resetn = 1'b1;
        repeat (6) @(posedge clk);

        mmio_read(32'h00, rd);
        $display("[1] ID = %08h %s", rd, (rd === 32'h54505531) ? "OK" : "FAIL");
        if (rd !== 32'h54505531) errors = errors + 1;

        // ---- 层配置 ----
        mmio_write(32'h0C, {16'd4, 16'd4});         // CFG_HW
        mmio_write(32'h10, {CIN[15:0], COUT[15:0]}); // CFG_CC: cin=8 cout=32
        mmio_write(32'h14, 32'h0000_0005);          // CFG_MD: stride1 pad1 k3 relu0 shift0
        $display("[2] 层配置 h=4 w=4 cin=%0d cout=%0d pad=1 stride=1", CIN, COUT);

        // ---- DMA + 权重 DMA 配置 ----
        mmio_write(32'h40, ACT_ADDR);
        mmio_write(32'h44, ACT_BYTES);
        mmio_write(32'h48, OUT_ADDR);
        mmio_write(32'h4C, OUT_BYTES);
        mmio_write(32'h50, 32'd1);                 // dma_en
        mmio_write(32'h58, WT_ADDR);                // R_WT_AB
        mmio_write(32'h5C, 32'd1);                  // R_WT_CFG: wt_en=1
        mmio_read(32'h5C, rd);
        if (rd[0] !== 1'b1) begin $display("  [FAIL] wt_en 未生效"); errors = errors + 1; end
        $display("[3] 权重 DMA 已启用，基址=%08h  总字节=%0d（每组 %0d）",
                 WT_ADDR, WT_BYTES, WT_BYTES/2);

        // ---- bias：start 之前写（引擎 S_IDLE 时 b_ready 恒 1）----
        for (co = 0; co < COUT; co = co + 1) begin
            wait_bfree();
            mmio_write(32'h24, co);                 // BADDR
            mmio_write(32'h20, 32'd0);              // BDATA = 0
        end
        $display("[4] bias %0d 条写毕（start 前）", COUT);

        // ---- start：之后 CPU 不再参与任何权重搬运 ----
        mmio_write(32'h04, 32'd1);
        $display("[5] start 已下发，CPU 进入纯等待");

        i = 0; finished = 1'b0;
        while (!finished && i < 4000) begin
            mmio_read(32'h08, rd);
            if (rd[1]) finished = 1'b1;             // bit1 = done
            i = i + 1;
        end
        mmio_read(32'h08, rd);
        $display("[6] 轮询 %0d 轮后 STATUS = %08h (bit1=done bit6=error)", i, rd);
        if (rd[1] !== 1'b1) begin $display("  [FAIL] 未见到 done"); errors = errors + 1; end
        if (rd[6] !== 1'b0) begin $display("  [FAIL] error 置起"); errors = errors + 1; end
        mmio_read(32'h38, rd);
        if (rd !== 0) begin $display("  [FAIL] ERR = %0d", rd); errors = errors + 1; end
        mmio_read(32'h34, rd);
        $display("    计算周期 CYC = %0d", rd);

        // ---- 校验输出 ----
        c32=0; c64=0; c48=0; c96=0; c72=0; c144=0; cother=0;
        for (i = 0; i < OUT_BYTES; i = i + 1) begin
            case (mem[OUT_ADDR + i])
                8'd32:  c32  = c32  + 1;
                8'd64:  c64  = c64  + 1;
                8'd48:  c48  = c48  + 1;
                8'd96:  c96  = c96  + 1;
                8'd72:  c72  = c72  + 1;
                8'd144: c144 = c144 + 1;
                default: cother = cother + 1;
            endcase
        end
        $display("[7] 输出统计: 32->%0d 64->%0d | 48->%0d 96->%0d | 72->%0d 144->%0d | 其它->%0d",
                 c32, c64, c48, c96, c72, c144, cother);
        $display("    期望      : 32->64  64->64  | 48->128 96->128 | 72->64  144->64 | 其它->0");
        if (c32 !== 64 || c64 !== 64 || c48 !== 128 || c96 !== 128 ||
            c72 !== 64 || c144 !== 64 || cother !== 0) begin
            $display("  [FAIL] 输出内容不符（lane->cout 映射或权重 DMA 出错）");
            errors = errors + 1;
            for (i = 0; i < 4; i = i + 1)
                $display("    px%0d ch0..15 = %02h %02h %02h %02h %02h %02h %02h %02h",
                         i, mem[OUT_ADDR+i*16+0], mem[OUT_ADDR+i*16+1],
                         mem[OUT_ADDR+i*16+2], mem[OUT_ADDR+i*16+3],
                         mem[OUT_ADDR+i*16+4], mem[OUT_ADDR+i*16+5],
                         mem[OUT_ADDR+i*16+6], mem[OUT_ADDR+i*16+7]);
        end else begin
            $display("  [OK] 输出与手算一致：偶通道权重=1、奇通道权重=2 被正确区分");
        end

        if (errors == 0) $display("\n==== tb_tpu_wdma PASS ====");
        else             $display("\n==== tb_tpu_wdma FAIL (%0d 处) ====", errors);
        $finish;
    end
endmodule
