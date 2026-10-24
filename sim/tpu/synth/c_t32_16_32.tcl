set_option -top t32_16_32
set_option -device PG2L200H-6FBB676
set_option -vlog_std verilog2001
add_file -verilog t32_16_32.v
add_file -verilog ../../../myriscv/tpu_conv_engine.v
run_compile
