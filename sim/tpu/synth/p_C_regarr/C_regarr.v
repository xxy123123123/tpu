
module C_regarr(input clk,input we,input signed [15:0] p0,input signed [15:0] p1,output reg [31:0] o);
reg signed [31:0] pacc [0:3]; integer i; reg signed [31:0] s;
always @(posedge clk) begin
  for(i=0;i<4;i=i+1) begin
    s = we ? 32'd0 : pacc[i];
    s = s + p0 + p1;
    pacc[i] <= s;
  end
  o <= pacc[0];
end
endmodule
