
module A_2dmem(input clk,input [1:0] wp,input [5:0] wa,input we,input [31:0] wd,input [1:0] rp,input [6:0] ra,output reg [31:0] rd);
reg [31:0] m [0:3][0:63]; integer i;
always @(posedge clk) begin
  for(i=0;i<4;i=i+1) if(we&&(i==wp)) m[i][wa]<=wd;
  rd<=m[rp][ra];
end
endmodule
