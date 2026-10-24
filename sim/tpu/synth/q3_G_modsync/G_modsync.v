module G_modsync(input clk,input we,input [1:0] wa,input [9:0] wi,input [31:0] wd,input [9:0] ra,output reg [127:0] rd);
reg [31:0] m [0:3][0:1023];
integer i;
always @(posedge clk) begin
    for(i=0;i<4;i=i+1) if(we&&(i==wa)) m[i][wi]<=wd;
end
generate
genvar gp;
for (gp=0; gp<4; gp=gp+1) begin : g
    always @(posedge clk) rd[gp*32 +: 32] <= m[gp][ra];
end
endgenerate
endmodule
