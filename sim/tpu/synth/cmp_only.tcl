set_option -top tpu_synth_wrap
set_option -device PG2L200H-6FBB676
set_option -vlog_std verilog2001
add_file -verilog ../tpu_synth_wrap.v
add_file -verilog ../../../myriscv/tpu_conv_engine.v
run_compile
