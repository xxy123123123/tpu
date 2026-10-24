
module D_genlane(input clk,input [1:0] wp,input [5:0] wa,input we,input [31:0] wd,input [5:0] ra,output [127:0] rd);
genvar gp;
generate
for (gp=0; gp<4; gp=gp+1) begin : g_w
    reg [31:0] wb [0:63];
    always @(posedge clk) if (we && (wp==gp)) wb[wa] <= wd;
    assign rd[gp*32 +: 32] = wb[ra];
end
endgenerate
endmodule
