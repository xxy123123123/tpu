set_option -top t16_16_128
set_option -device PG2L200H-6FBB676
set_option -vlog_std verilog2001
add_file -verilog t16_16_128.v
add_file -verilog ../../../myriscv/tpu_conv_engine.v
run_compile
