`timescale 1ns/1ps
// TPU SoC 级软硬件协同仿真：
//   真实 CPU(mycpu_sync) 从 DDR 模型启动 C 程序(tests/tpu_conv)，
//   经 data_bus_interconnect 的 0x1000_5000 窗口访问 tpu_mmio；
//   TPU DMA 的权重/激活读、输出写与 CPU 共享同一个扁平 AXI 内存模型。
// C 程序自检并在 UART 打印 TPU_SOC_PASS / TPU_SOC_FAIL；本 TB 监听 UART 判决。
//
// 内存模型地址单位（与 myriscv 内部约定一致）：
//   CPU 口(x16)：axi_addr = 字节地址 >> 1，每 beat 16 字节
//   TPU 口     ：axi_addr = 字节地址 >> 2，每 beat 32 字节，单 beat（rlast 恒 1）
module tb_tpu_soc;
    localparam P = 16, C = 8;
    localparam integer MEM_WORDS = 1051136;   // 256bit 字，覆盖 0..0x0201_3FFF

    reg clk = 0, resetn = 0;
    always #(1.0e9 / 93750000.0 / 2.0) clk = ~clk;

    // ================= CPU SoC =================
    wire [27:0]  cpu_awaddr, cpu_araddr;
    wire         cpu_awvalid, cpu_arvalid, cpu_awready, cpu_arready;
    wire [127:0] cpu_wdata;
    wire [15:0]  cpu_wstrb;
    wire         cpu_wready;
    reg  [127:0] cpu_rdata = 0;
    reg          cpu_rvalid = 0;
    wire [1:0]   selftest_status;
    wire         uart_tx;

    // TPU MMIO 从机口（soc_top 引出）
    wire        tpu_req_valid, tpu_req_write, tpu_req_ready;
    wire [1:0]  tpu_req_size;
    wire [31:0] tpu_req_addr, tpu_req_wdata;
    wire [3:0]  tpu_req_wstrb;
    wire        tpu_rsp_valid, tpu_rsp_error;
    wire [31:0] tpu_rsp_rdata;

    soc_top #(
        .ENABLE_DDR(1), .ENABLE_TPU(1), .ENABLE_PREPROCESS(1),
        .RESET_PC(32'h4000_0000), .INST_ROM_BASE(0), .DATA_RAM_BASE(0),
        .UART_RSP_DELAY_CYCLES(7)
    ) dut (
        .clk(clk), .resetn(resetn),
        .irq_external(1'b0), .irq_software(1'b0), .irq_timer(1'b0),
        .cnn_req_ready(1'b0), .cnn_rsp_valid(1'b0), .cnn_rsp_error(1'b0), .cnn_rsp_rdata(32'd0),
        .cnn_req_valid(), .cnn_req_write(), .cnn_req_size(), .cnn_req_addr(),
        .cnn_req_wdata(), .cnn_req_wstrb(),
        .pp_req_ready(1'b0), .pp_rsp_valid(1'b0), .pp_rsp_error(1'b0), .pp_rsp_rdata(32'd0),
        .pp_req_valid(), .pp_req_write(), .pp_req_size(), .pp_req_addr(),
        .pp_req_wdata(), .pp_req_wstrb(),
        .preprocess_busy(1'b0), .preprocess_done(1'b0),
        .cam_req_ready(1'b0), .cam_rsp_valid(1'b0), .cam_rsp_error(1'b0), .cam_rsp_rdata(32'd0),
        .cam_req_valid(), .cam_req_write(), .cam_req_size(), .cam_req_addr(),
        .cam_req_wdata(), .cam_req_wstrb(),
        .camera_busy(1'b0), .camera_dual_mode(1'b0), .camera_cpu_owned(2'b00),
        .tpu_req_valid(tpu_req_valid), .tpu_req_write(tpu_req_write),
        .tpu_req_size(tpu_req_size), .tpu_req_addr(tpu_req_addr),
        .tpu_req_wdata(tpu_req_wdata), .tpu_req_wstrb(tpu_req_wstrb),
        .tpu_req_ready(tpu_req_ready), .tpu_rsp_valid(tpu_rsp_valid),
        .tpu_rsp_error(tpu_rsp_error), .tpu_rsp_rdata(tpu_rsp_rdata),
        .debug_wb_pc(), .debug_wb_rf_we(), .debug_wb_rf_wnum(),
        .debug_wb_rf_wdata(), .debug_inst(),
        .ddr_init_done(1'b1),
        .ddr_axi_awaddr(cpu_awaddr), .ddr_axi_awuser_ap(), .ddr_axi_awuser_id(),
        .ddr_axi_awlen(), .ddr_axi_awready(cpu_awready), .ddr_axi_awvalid(cpu_awvalid),
        .ddr_axi_wdata(cpu_wdata), .ddr_axi_wstrb(cpu_wstrb), .ddr_axi_wready(cpu_wready),
        .ddr_axi_wusero_id(4'd0), .ddr_axi_wusero_last(cpu_wready),
        .ddr_axi_araddr(cpu_araddr), .ddr_axi_aruser_ap(), .ddr_axi_aruser_id(),
        .ddr_axi_arlen(), .ddr_axi_arready(cpu_arready), .ddr_axi_arvalid(cpu_arvalid),
        .ddr_axi_rdata(cpu_rdata), .ddr_axi_rid(4'd0),
        .ddr_axi_rlast(cpu_rvalid), .ddr_axi_rvalid(cpu_rvalid),
        .selftest_status(selftest_status),
        .uart_rx(1'b1), .uart_tx(uart_tx)
    );

    // ================= TPU 栈（同 soc_ddr3_top 的结构）=================
    wire [27:0]  act_araddr, tpu_awaddr;
    wire         act_arvalid, tpu_awvalid;
    wire [255:0] tpu_wdata;
    wire         act_arready, tpu_awready, tpu_wready;
    wire         act_rvalid;
    wire [3:0]   act_arid;
    wire [27:0]  wt_araddr;
    wire         wt_arvalid, wt_arready, wt_rvalid;
    wire [3:0]   wt_rid;
    wire         act_rerr, wt_rerr;
    wire         tpu_busy, tpu_done, tpu_pg_req;

    tpu_mmio #(.P(P), .C(C)) u_tpu (
        .clk(clk), .resetn(resetn),
        .req_valid(tpu_req_valid), .req_write(tpu_req_write),
        .req_size(tpu_req_size), .req_addr(tpu_req_addr),
        .req_wdata(tpu_req_wdata), .req_wstrb(tpu_req_wstrb),
        .req_ready(tpu_req_ready), .rsp_valid(tpu_rsp_valid),
        .rsp_error(tpu_rsp_error), .rsp_rdata(tpu_rsp_rdata),
        .araddr(act_araddr), .awaddr(tpu_awaddr),
        .arvalid(act_arvalid), .awvalid(tpu_awvalid), .wdata(tpu_wdata),
        .arready(act_arready), .rvalid(act_rvalid), .rlast(1'b1),
        .awready(tpu_awready), .wready(tpu_wready),
        .arid(act_arid), .rdata(slv_rdata), .read_protocol_error(act_rerr),
        .wt_araddr(wt_araddr), .wt_arvalid(wt_arvalid), .wt_arready(wt_arready),
        .wt_rvalid(wt_rvalid), .wt_rlast(1'b1), .wt_rid(wt_rid),
        .wt_rdata(slv_rdata), .wt_read_protocol_error(wt_rerr),
        .tpu_busy(tpu_busy), .tpu_done(tpu_done), .tpu_pg_req(tpu_pg_req)
    );

    // 两条 TPU 读口经真实仲裁器汇聚（与 soc_ddr3_top 同构；手写仲裁踩过相位死锁）
    wire [27:0]  slv_araddr;
    wire [3:0]   slv_arid;
    wire         slv_arvalid, slv_arready;
    reg          slv_rvalid = 0;
    reg  [255:0] slv_rdata = 0;
    reg  [3:0]   slv_rid = 0;
    assign act_arid = slv_rid;
    assign wt_rid   = slv_rid;
    assign act_rerr = 1'b0;

    preprocess_ddr_read_arbiter #(.PP_ID(4'd6)) u_wt_read_arbiter (
        .clk(clk), .resetn(resetn),
        .cpu_addr(act_araddr), .cpu_len(4'd0), .cpu_id(4'd5), .cpu_ap(1'b0),
        .cpu_valid(act_arvalid), .cpu_ready(act_arready), .cpu_rvalid(act_rvalid),
        .pp_addr(wt_araddr), .pp_valid(wt_arvalid), .pp_ready(wt_arready), .pp_rvalid(wt_rvalid),
        .araddr(slv_araddr), .arlen(), .arid(slv_arid), .arap(),
        .arvalid(slv_arvalid), .arready(slv_arready),
        .rvalid(slv_rvalid), .rlast(1'b1), .rid(slv_rid),
        .protocol_error(wt_rerr)
    );

    // ================= 共享扁平内存模型 =================
    reg [255:0] mem [0:MEM_WORDS-1];
    reg [31:0]  image [0:65535];
    string      dat_path;
    integer     i, img_words;

    initial begin
        if (!$value$plusargs("DAT=%s", dat_path)) $fatal(1, "RESULT: FAIL missing +DAT");
        $readmemh(dat_path, image);
        img_words = 0;
        // 以文件尾前首个非 x 字为准估计镜像长度
        for (i = 65535; i >= 0; i = i - 1)
            if (img_words == 0 && image[i] !== 32'hxxxx_xxxx) img_words = i + 1;
        for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = {8{32'ha5a5_a5a5}};
        for (i = 0; i < img_words; i = i + 1) mem[i/8][(i%8)*32 +: 32] = image[i];
    end

    // ---- CPU 读口（x16，beat = 16 字节）----
    reg        cpu_reading = 0;
    reg [27:0] cpu_ra = 0;
    integer    cpu_ar_wait = 0, cpu_r_wait = 0;
    assign cpu_arready = cpu_arvalid && !cpu_reading && (cpu_ar_wait >= 3);
    always @(posedge clk) begin
        cpu_rvalid <= 0;
        if (resetn) begin
            if (cpu_arvalid && !cpu_arready) cpu_ar_wait <= cpu_ar_wait + 1;
            if (cpu_arvalid && cpu_arready) begin
                cpu_ra <= cpu_araddr; cpu_reading <= 1; cpu_r_wait <= 4; cpu_ar_wait <= 0;
            end
            if (cpu_reading && cpu_r_wait > 0) cpu_r_wait <= cpu_r_wait - 1;
            if (cpu_reading && cpu_r_wait == 0) begin
                cpu_rdata <= cpu_ra[3] ? mem[cpu_ra >> 4][255:128]
                                       : mem[cpu_ra >> 4][127:0];
                cpu_rvalid <= 1; cpu_reading <= 0;
            end
        end
    end

    // ---- CPU 写口（x16，带 wstrb）----
    reg        cpu_writing = 0;
    reg [27:0] cpu_wa = 0;
    integer    cpu_aw_wait = 0, cpu_w_wait = 0;
    integer    lane;
    assign cpu_awready = cpu_awvalid && !cpu_writing && (cpu_aw_wait >= 2);
    assign cpu_wready  = cpu_writing && (cpu_w_wait == 0);
    always @(posedge clk) begin
        if (resetn) begin
            if (cpu_awvalid && !cpu_awready) cpu_aw_wait <= cpu_aw_wait + 1;
            if (cpu_awvalid && cpu_awready) begin
                cpu_wa <= cpu_awaddr; cpu_writing <= 1; cpu_w_wait <= 3; cpu_aw_wait <= 0;
            end
            if (cpu_writing && cpu_w_wait > 0) cpu_w_wait <= cpu_w_wait - 1;
            if (cpu_wready) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    if (cpu_wstrb[lane])
                        mem[cpu_wa >> 4][cpu_wa[3]*128 + lane*8 +: 8] <= cpu_wdata[lane*8 +: 8];
                cpu_writing <= 0;
            end
        end
    end

    // ---- TPU 读口（仲裁器下游，256bit，单 beat）----
    // rid 必须与 rvalid 同拍可见，否则仲裁器判协议错误并永久锁死。
    reg        rd_pend = 0;
    reg [31:0] rd_word = 0;
    reg [3:0]  rid_hold = 0;
    reg        adiv = 0;
    integer    k;
    assign slv_arready = adiv;
    always @(posedge clk) begin
        if (!resetn) begin
            rd_pend <= 0; slv_rvalid <= 0; rid_hold <= 0; adiv <= 0;
        end else begin
            adiv <= ~adiv;
            slv_rvalid <= 0;
            if (rd_pend) begin
                rd_pend <= 0;
                slv_rvalid <= 1'b1;
                slv_rid <= rid_hold;
                for (k = 0; k < 32; k = k + 1)
                    slv_rdata[k*8 +: 8] = mem[rd_word][k*8 +: 8];
            end else if (slv_arvalid && slv_arready) begin
                rd_word  <= slv_araddr >> 3;
                rid_hold <= slv_arid;
                slv_rid  <= slv_arid;
                rd_pend  <= 1'b1;
            end
        end
    end

    // ---- TPU 写口（256bit，无 wstrb，整 beat 写）----
    reg        wr_armed = 0;
    reg [31:0] wr_word = 0;
    reg        wdiv = 0;
    assign tpu_awready = 1'b1;
    assign tpu_wready  = wdiv;
    always @(posedge clk) begin
        if (!resetn) begin
            wdiv <= 0; wr_armed <= 0; wr_word <= 0;
        end else begin
            wdiv <= ~wdiv;
            if (tpu_awvalid && tpu_awready) begin
                wr_word  <= tpu_awaddr >> 3;
                wr_armed <= 1'b1;
            end
            if (wr_armed && tpu_wready) begin
                mem[wr_word] <= tpu_wdata;
                wr_armed <= 1'b0;
            end
        end
    end

    // ================= UART 监听与判决 =================
    wire       tx_fire = dut.u_uart_mmio.tx_send;
    wire [7:0] tx_byte = dut.u_uart_mmio.req_wdata[7:0];
    reg [8*13-1:0] tail = 0;
    integer cycles = 0;

    always @(posedge clk) begin
        if (tx_fire) begin
            $write("%c", tx_byte);
            tail = {tail[8*12-1:0], tx_byte};
            if (tail == "TPU_SOC_PASS\n") begin
                $display("\nRESULT: PASS tpu_soc cycles=%0d", cycles);
                $finish;
            end
            if (tail[8*13-1:8] == "TPU_SOC_FAIL") begin
                $display("\nRESULT: FAIL tpu_soc (C 自检未过)");
                $fatal(1);
            end
        end
    end

    // ================= 复位与超时 =================
    initial begin
        repeat (20) @(posedge clk);
        resetn = 1'b1;
    end
    always @(posedge clk) begin
        cycles <= cycles + 1;
`ifdef TPU_SOC_DEBUG
        if (cycles > 200000) begin
            $display("\n[dbg] pc=%08h tpu_req v=%b rdy=%b rsp_v=%b",
                     dut.debug_wb_pc, tpu_req_valid, tpu_req_ready, tpu_rsp_valid);
            $fatal(1, "RESULT: FAIL tpu_soc debug stop");
        end
`else
        // The firmware prints all performance counters and the output histogram
        // through the cycle-accurate 115200-baud UART before the final verdict.
        if (cycles > 4000000) $fatal(1, "RESULT: FAIL tpu_soc timeout");
`endif
    end
`ifdef TPU_SOC_DEBUG
    always @(posedge clk) begin
        if (tpu_req_valid)
            $display("[dbg %0d] tpu_req wr=%b addr=%02h wdata=%08h rdy=%b",
                     cycles, tpu_req_write, tpu_req_addr[7:0], tpu_req_wdata, tpu_req_ready);
        if (tpu_rsp_valid)
            $display("[dbg %0d] tpu_rsp rdata=%08h err=%b", cycles, tpu_rsp_rdata, tpu_rsp_error);
    end
`endif
endmodule
