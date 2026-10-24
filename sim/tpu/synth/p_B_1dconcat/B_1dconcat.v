
module B_1dconcat(input clk,input [2:0] b,input [1:0] cb,input [5:0] fx,input we,input [31:0] wd,output reg [31:0] rd);
reg [31:0] m [0:5*4*64-1];
always @(posedge clk) begin
  if(we) m[{b,cb,fx}]<=wd;
  rd<=m[{b,cb,fx}];
end
endmodule
