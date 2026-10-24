
module E_loopread(input clk,input [5:0] ra,output reg [127:0] rd);
reg [31:0] m [0:255]; integer i;
always @(posedge clk) begin
  for(i=0;i<4;i=i+1) rd[i*32 +: 32] <= m[{i[1:0],ra}];
end
endmodule
