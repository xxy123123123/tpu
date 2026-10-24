# TPU 布局布线评估（P32C16, 200MHz 约束）：IO 收窄顶层，跑通 dev_map -> pnr -> report_timing
# 目的：拿到布局后的真实 Fmax 与关键路径，指导 P0 时序优化
add_design "D:/riscv/RISCV/sim/tpu/synth/tpu_pnr_wrap_v9.v"
add_design "D:/riscv/RISCV/myriscv/tpu_conv_engine_v9.v"
add_constraint "D:/riscv/RISCV/sim/tpu/synth/tpu_pnr.fdc"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module tpu_pnr_wrap_v9
synthesize -ads -selected_syn_tool_opt 2
dev_map
pnr -gplace_seed 3 -gplace_times 1 -parallel 4 -mode {performance}
report_timing
