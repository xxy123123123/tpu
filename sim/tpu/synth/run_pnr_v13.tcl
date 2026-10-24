# TPU 布局布线评估 v13：P32C32（MAC 阵列 1024/cycle，APM 预期 256）
# 目的：验证「以面积换吞吐」——C 从 16 提到 32 后 Fmax 掉多少、资源是否装得下
add_design "D:/riscv/RISCV/sim/tpu/synth/tpu_pnr_wrap_v13.v"
add_design "D:/riscv/RISCV/myriscv/tpu_conv_engine.v"
add_constraint "D:/riscv/RISCV/sim/tpu/synth/tpu_pnr.fdc"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module tpu_pnr_wrap_v13
synthesize -ads -selected_syn_tool_opt 2
dev_map
pnr -gplace_seed 2 -gplace_times 2 -parallel 4 -mode {performance}
report_timing
