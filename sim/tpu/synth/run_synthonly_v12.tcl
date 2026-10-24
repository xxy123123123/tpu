add_design "D:/riscv/RISCV/sim/tpu/synth/tpu_pnr_wrap_v12.v"
add_design "D:/riscv/RISCV/myriscv/tpu_conv_engine_v12.v"
add_constraint "D:/riscv/RISCV/sim/tpu/synth/tpu_pnr.fdc"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module tpu_pnr_wrap_v12
synthesize -ads -selected_syn_tool_opt 2
