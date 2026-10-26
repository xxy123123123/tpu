`timescale 1ns/1ps
// data_bus_interconnect 的 TPU 窗口译码验证
//   1. 0x1000_500x 命中 TPU，且送给从设备的地址已减基址
//   2. 0x1000_400x 命中 CNN 而非 TPU
//   3. 0x8000_0000 命中 RAM 而非 TPU
module tb_bus_tpu;
    reg clk = 0, resetn = 0;

    wire cnn_req_valid;
    wire cnn_req_write;
    wire [1:0] cnn_req_size;
    wire [31:0] cnn_req_addr;
    wire [31:0] cnn_req_wdata;
    wire [3:0] cnn_req_wstrb;
    reg  cnn_req_ready;
    reg  cnn_rsp_valid;
    reg  cnn_rsp_error;
    reg  [31:0] cnn_rsp_rdata;
    wire pp_req_valid;
    wire pp_req_write;
    wire [1:0] pp_req_size;
    wire [31:0] pp_req_addr;
    wire [31:0] pp_req_wdata;
    wire [3:0] pp_req_wstrb;
    reg  pp_req_ready;
    reg  pp_rsp_valid;
    reg  pp_rsp_error;
    reg  [31:0] pp_rsp_rdata;
    reg  preprocess_busy;
    reg  preprocess_done;
    wire cam_req_valid;
    wire cam_req_write;
    wire [1:0] cam_req_size;
    wire [31:0] cam_req_addr;
    wire [31:0] cam_req_wdata;
    wire [3:0] cam_req_wstrb;
    reg  cam_req_ready;
    reg  cam_rsp_valid;
    reg  cam_rsp_error;
    reg  [31:0] cam_rsp_rdata;
    reg  camera_busy;
    wire tpu_req_valid;
    wire tpu_req_write;
    wire [1:0] tpu_req_size;
    wire [31:0] tpu_req_addr;
    wire [31:0] tpu_req_wdata;
    wire [3:0] tpu_req_wstrb;
    reg  tpu_req_ready;
    reg  tpu_rsp_valid;
    reg  tpu_rsp_error;
    reg  [31:0] tpu_rsp_rdata;
    reg  camera_dual_mode;
    reg  [1:0] camera_cpu_owned;
    reg  ddr_write_buffer_busy;
    reg  m_req_valid;
    reg  m_req_write;
    reg  [ 1:0] m_req_size;
    reg  [31:0] m_req_addr;
    reg  [31:0] m_req_wdata;
    reg  [ 3:0] m_req_wstrb;
    wire m_req_ready;
    wire m_rsp_valid;
    wire [31:0] m_rsp_rdata;
    wire m_rsp_error;
    wire ram_req_valid;
    wire ram_req_write;
    wire [ 1:0] ram_req_size;
    wire [31:0] ram_req_addr;
    wire [31:0] ram_req_wdata;
    wire [ 3:0] ram_req_wstrb;
    reg  ram_req_ready;
    reg  ram_rsp_valid;
    reg  [31:0] ram_rsp_rdata;
    reg  ram_rsp_error;
    wire mmio_req_valid;
    wire mmio_req_write;
    wire [ 1:0] mmio_req_size;
    wire [31:0] mmio_req_addr;
    wire [31:0] mmio_req_wdata;
    wire [ 3:0] mmio_req_wstrb;
    reg  mmio_req_ready;
    reg  mmio_rsp_valid;
    reg  [31:0] mmio_rsp_rdata;
    reg  mmio_rsp_error;
    wire ddr_req_valid;
    wire ddr_req_write;
    wire [ 1:0] ddr_req_size;
    wire [31:0] ddr_req_addr;
    wire [31:0] ddr_req_wdata;
    wire [ 3:0] ddr_req_wstrb;
    reg  ddr_req_ready;
    reg  ddr_rsp_valid;
    reg  [31:0] ddr_rsp_rdata;
    reg  ddr_rsp_error;
    wire uart_req_valid;
    wire uart_req_write;
    wire [ 1:0] uart_req_size;
    wire [31:0] uart_req_addr;
    wire [31:0] uart_req_wdata;
    wire [ 3:0] uart_req_wstrb;
    reg  uart_req_ready;
    reg  uart_rsp_valid;
    reg  [31:0] uart_rsp_rdata;
    reg  uart_rsp_error;

    data_bus_interconnect #(
        .RAM_BASE(32'h8000_0000), .RAM_ADDR_MASK(32'hffff_c000),
        .MMIO_BASE(32'h1000_0000), .MMIO_ADDR_MASK(32'hffff_f000),
        .ENABLE_DDR(0),
        .ENABLE_CNN(1), .ENABLE_PREPROCESS(0), .ENABLE_CAMERA(0),
        .ENABLE_TPU(1), .ENABLE_UART(0),
        .UART_BASE(32'h1000_1000)
    ) u_dut (
        .clk(clk), .resetn(resetn),
        .cnn_req_valid(cnn_req_valid),
        .cnn_req_write(cnn_req_write),
        .cnn_req_size(cnn_req_size),
        .cnn_req_addr(cnn_req_addr),
        .cnn_req_wdata(cnn_req_wdata),
        .cnn_req_wstrb(cnn_req_wstrb),
        .cnn_req_ready(cnn_req_ready),
        .cnn_rsp_valid(cnn_rsp_valid),
        .cnn_rsp_error(cnn_rsp_error),
        .cnn_rsp_rdata(cnn_rsp_rdata),
        .pp_req_valid(pp_req_valid),
        .pp_req_write(pp_req_write),
        .pp_req_size(pp_req_size),
        .pp_req_addr(pp_req_addr),
        .pp_req_wdata(pp_req_wdata),
        .pp_req_wstrb(pp_req_wstrb),
        .pp_req_ready(pp_req_ready),
        .pp_rsp_valid(pp_rsp_valid),
        .pp_rsp_error(pp_rsp_error),
        .pp_rsp_rdata(pp_rsp_rdata),
        .preprocess_busy(preprocess_busy),
        .preprocess_done(preprocess_done),
        .cam_req_valid(cam_req_valid),
        .cam_req_write(cam_req_write),
        .cam_req_size(cam_req_size),
        .cam_req_addr(cam_req_addr),
        .cam_req_wdata(cam_req_wdata),
        .cam_req_wstrb(cam_req_wstrb),
        .cam_req_ready(cam_req_ready),
        .cam_rsp_valid(cam_rsp_valid),
        .cam_rsp_error(cam_rsp_error),
        .cam_rsp_rdata(cam_rsp_rdata),
        .camera_busy(camera_busy),
        .tpu_req_valid(tpu_req_valid),
        .tpu_req_write(tpu_req_write),
        .tpu_req_size(tpu_req_size),
        .tpu_req_addr(tpu_req_addr),
        .tpu_req_wdata(tpu_req_wdata),
        .tpu_req_wstrb(tpu_req_wstrb),
        .tpu_req_ready(tpu_req_ready),
        .tpu_rsp_valid(tpu_rsp_valid),
        .tpu_rsp_error(tpu_rsp_error),
        .tpu_rsp_rdata(tpu_rsp_rdata),
        .camera_dual_mode(camera_dual_mode),
        .camera_cpu_owned(camera_cpu_owned),
        .ddr_write_buffer_busy(ddr_write_buffer_busy),
        .m_req_valid(m_req_valid),
        .m_req_write(m_req_write),
        .m_req_size(m_req_size),
        .m_req_addr(m_req_addr),
        .m_req_wdata(m_req_wdata),
        .m_req_wstrb(m_req_wstrb),
        .m_req_ready(m_req_ready),
        .m_rsp_valid(m_rsp_valid),
        .m_rsp_rdata(m_rsp_rdata),
        .m_rsp_error(m_rsp_error),
        .ram_req_valid(ram_req_valid),
        .ram_req_write(ram_req_write),
        .ram_req_size(ram_req_size),
        .ram_req_addr(ram_req_addr),
        .ram_req_wdata(ram_req_wdata),
        .ram_req_wstrb(ram_req_wstrb),
        .ram_req_ready(ram_req_ready),
        .ram_rsp_valid(ram_rsp_valid),
        .ram_rsp_rdata(ram_rsp_rdata),
        .ram_rsp_error(ram_rsp_error),
        .mmio_req_valid(mmio_req_valid),
        .mmio_req_write(mmio_req_write),
        .mmio_req_size(mmio_req_size),
        .mmio_req_addr(mmio_req_addr),
        .mmio_req_wdata(mmio_req_wdata),
        .mmio_req_wstrb(mmio_req_wstrb),
        .mmio_req_ready(mmio_req_ready),
        .mmio_rsp_valid(mmio_rsp_valid),
        .mmio_rsp_rdata(mmio_rsp_rdata),
        .mmio_rsp_error(mmio_rsp_error),
        .ddr_req_valid(ddr_req_valid),
        .ddr_req_write(ddr_req_write),
        .ddr_req_size(ddr_req_size),
        .ddr_req_addr(ddr_req_addr),
        .ddr_req_wdata(ddr_req_wdata),
        .ddr_req_wstrb(ddr_req_wstrb),
        .ddr_req_ready(ddr_req_ready),
        .ddr_rsp_valid(ddr_rsp_valid),
        .ddr_rsp_rdata(ddr_rsp_rdata),
        .ddr_rsp_error(ddr_rsp_error),
        .uart_req_valid(uart_req_valid),
        .uart_req_write(uart_req_write),
        .uart_req_size(uart_req_size),
        .uart_req_addr(uart_req_addr),
        .uart_req_wdata(uart_req_wdata),
        .uart_req_wstrb(uart_req_wstrb),
        .uart_req_ready(uart_req_ready),
        .uart_rsp_valid(uart_rsp_valid),
        .uart_rsp_rdata(uart_rsp_rdata),
        .uart_rsp_error(uart_rsp_error)
    );

    always #5 clk = ~clk;

    integer errors = 0;
    wire [2:0] hits;
    assign hits = {tpu_req_valid, cnn_req_valid, ram_req_valid};

    initial begin
        ram_req_ready = 1; mmio_req_ready = 1; uart_req_ready = 1;
        cam_req_ready = 1; pp_req_ready = 1; cnn_req_ready = 1; tpu_req_ready = 1;
        ddr_req_ready = 1;
        ram_rsp_valid = 0; mmio_rsp_valid = 0; uart_rsp_valid = 0;
        cam_rsp_valid = 0; pp_rsp_valid = 0; cnn_rsp_valid = 0; tpu_rsp_valid = 0;
        ddr_rsp_valid = 0;
        camera_busy = 0; camera_dual_mode = 0; camera_cpu_owned = 0;
        preprocess_busy = 0; preprocess_done = 0; ddr_write_buffer_busy = 0;
        m_req_valid = 0; m_req_write = 0; m_req_addr = 0; m_req_wdata = 0;
        m_req_size = 0; m_req_wstrb = 0;

        repeat (4) @(posedge clk);
        resetn = 1'b1;
        repeat (4) @(posedge clk);

        $display("[1] TPU 窗口 0x1000_5000 应命中 TPU");
        probe(32'h1000_5004, 3'b100, "TPU+4 ");
        probe(32'h1000_5000, 3'b100, "TPU+0 ");
        probe(32'h1000_50fc, 3'b100, "TPU+fc");

        $display("[2] 其它窗口不应误选 TPU");
        probe(32'h1000_4004, 3'b010, "CNN   ");
        probe(32'h8000_0000, 3'b001, "RAM   ");

        $display("[3] 送给 TPU 的地址须已减基址");
        m_req_valid <= 1'b1; m_req_write <= 1'b1;
        m_req_addr <= 32'h1000_502c; m_req_size <= 2'd2; m_req_wstrb <= 4'hf;
        @(posedge clk);
        if (tpu_req_addr !== 32'h0000_002c) begin
            $display("  [FAIL] tpu_req_addr=%08h (期望 0000002c)", tpu_req_addr);
            errors = errors + 1;
        end else begin
            $display("  [OK]   tpu_req_addr=%08h", tpu_req_addr);
        end
        m_req_valid <= 1'b0;
        @(posedge clk);

        if (errors == 0) $display("\n==== tb_bus_tpu PASS ====");
        else             $display("\n==== tb_bus_tpu FAIL (%0d 处) ====", errors);
        $finish;
    end

    task probe(input [31:0] addr, input [2:0] want, input [63:0] tag);
        begin
            m_req_valid <= 1'b1; m_req_write <= 1'b1;
            m_req_addr  <= addr; m_req_wdata <= 32'hdead_beef;
            m_req_size  <= 2'd2; m_req_wstrb <= 4'hf;
            @(posedge clk);
            if (hits !== want) begin
                $display("  [FAIL] %0s addr=%08h -> tpu/cnn/ram=%b (期望 %b)",
                         tag, addr, hits, want);
                errors = errors + 1;
            end else begin
                $display("  [OK]   %0s addr=%08h -> tpu/cnn/ram=%b", tag, addr, hits);
            end
            // 被选中的从设备给一拍响应，否则 pending 一直占着，后续请求全被阻塞
            if (tpu_req_valid)  tpu_rsp_valid  = 1;
            if (cnn_req_valid)  cnn_rsp_valid  = 1;
            if (ram_req_valid)  ram_rsp_valid  = 1;
            if (mmio_req_valid) mmio_rsp_valid = 1;
            m_req_valid <= 1'b0;
            @(posedge clk);
            tpu_rsp_valid = 0; cnn_rsp_valid = 0;
            ram_rsp_valid = 0; mmio_rsp_valid = 0;
            @(posedge clk);
        end
    endtask
endmodule
