# TPU 布局布线评估 v15
add_design "D:/riscv/RISCV/sim/tpu/synth/tpu_pnr_wrap_v15.v"
add_design "D:/riscv/RISCV/myriscv/tpu_conv_engine.v"
add_constraint "D:/riscv/RISCV/sim/tpu/synth/tpu_pnr.fdc"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module tpu_pnr_wrap_v15
synthesize -ads -selected_syn_tool_opt 2
dev_map
pnr -gplace_seed 2 -gplace_times 2 -parallel 4 -mode {performance}
report_timing
