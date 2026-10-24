module P_gen_d512(input clk,input we,input [1:0] wa,input [8:0] wi,input [127:0] wd,input [8:0] ra,input gz,output reg [511:0] rd);
genvar gp;
generate
for (gp=0; gp<4; gp=gp+1) begin : g
    reg [127:0] m [0:511];
    always @(posedge clk) if (we && (wa==gp)) m[wi] <= wd;
    always @(posedge clk) if (!gz) rd[gp*128 +: 128] <= m[ra];
end
endgenerate
endmodule
