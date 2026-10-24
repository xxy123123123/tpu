add_design "o16_8_128.v"
add_design "D:/riscv/RISCV/myriscv/tpu_conv_engine.v"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module o16_8_128
