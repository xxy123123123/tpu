module probe_mem2 (
    input  wire clk,
    input  wire [4:0]  wp,
    input  wire [7:0]  wa,
    input  wire        we,
    input  wire [127:0] wd,
    input  wire [4:0]  rp,
    input  wire [8:0]  ra,
    output reg [127:0] rd
);
    reg [127:0] m [0:31][0:255];
    integer i;
    always @(posedge clk) begin
        for (i = 0; i < 32; i = i + 1)
            if (we && (i == wp)) m[i][wa] <= wd;
        rd <= m[rp][ra];
    end
endmodule
