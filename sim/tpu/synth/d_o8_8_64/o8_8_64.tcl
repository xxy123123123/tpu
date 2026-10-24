add_design "o8_8_64.v"
add_design "D:/riscv/RISCV/myriscv/tpu_conv_engine.v"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module o8_8_64
