# TPU 综合评估（P32C16, 200MHz）：跳过 dev_map（测试顶层 IO 裸露导致 IO DRC 超限，SoC 内为内部信号）
add_design "D:/riscv/RISCV/sim/tpu/synth/tpu_synth_wrap.v"
add_design "D:/riscv/RISCV/myriscv/tpu_conv_engine.v"
add_constraint "D:/riscv/RISCV/sim/tpu/synth/tpu.fdc"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module tpu_synth_wrap
synthesize -ads -selected_syn_tool_opt 2
report_timing
