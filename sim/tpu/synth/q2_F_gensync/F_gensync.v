module F_gensync(input clk,input we,input [1:0] wa,input [31:0] wd,input [9:0] ra,output reg [127:0] rd);
genvar gp;
generate
for (gp=0; gp<4; gp=gp+1) begin : g
    reg [31:0] m [0:1023];
    always @(posedge clk) if (we && (wa==gp)) m[wa] <= wd;
    always @(posedge clk) rd[gp*32 +: 32] <= m[ra];
end
endgenerate
endmodule
