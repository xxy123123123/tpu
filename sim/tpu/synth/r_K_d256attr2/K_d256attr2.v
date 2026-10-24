module K_d256attr2(input clk,input we,input [8:0] wi,input [127:0] wd,input [8:0] ra,output reg [127:0] rd);
(* ram_style = "block" *)
reg [127:0] m [0:255];
always @(posedge clk) if (we) m[wi] <= wd;
always @(posedge clk) rd <= m[ra];
endmodule
