# TPU 布局布线评估 v14
add_design "D:/riscv/RISCV/sim/tpu/synth/tpu_pnr_wrap_v14.v"
add_design "D:/riscv/RISCV/sim/tpu/synth/tpu_conv_engine_pnr_snap.v"
add_constraint "D:/riscv/RISCV/sim/tpu/synth/tpu_pnr.fdc"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module tpu_pnr_wrap_v14
synthesize -ads -selected_syn_tool_opt 2
dev_map
pnr -gplace_seed 2 -gplace_times 2 -parallel 4 -mode {performance}
report_timing
