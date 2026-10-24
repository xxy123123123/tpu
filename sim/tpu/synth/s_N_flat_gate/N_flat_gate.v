module N_flat_gate(input clk,input we,input [8:0] wi,input [127:0] wd,input [8:0] ra,input gz,output reg [127:0] rd);
reg [127:0] m [0:255];
always @(posedge clk) if (we) m[wi] <= wd;
always @(posedge clk) if (!gz) rd <= m[ra];
endmodule
