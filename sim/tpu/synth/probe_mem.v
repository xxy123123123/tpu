// 存储器推断探针：验证 PDS 能否推断非 2 的幂深度的同步读 RAM
module probe_mem (
    input  wire             clk,
    input  wire [13:0]      wa,
    input  wire             we,
    input  wire [127:0]     wd,
    input  wire [13:0]      ra,
    output reg  [127:0]     rd
);
`ifdef PROBE_POW2
    reg [127:0] m [0:16383];    // 2^14
`else
    reg [127:0] m [0:10239];    // 非 2 的幂
`endif
    always @(posedge clk) begin
        if (we) m[wa] <= wd;
        rd <= m[ra];
    end
endmodule
