set_option -top probe_mem2
set_option -device PG2L200H-6FBB676
set_option -vlog_std verilog2001
add_file -verilog probe_mem2.v
run_compile
