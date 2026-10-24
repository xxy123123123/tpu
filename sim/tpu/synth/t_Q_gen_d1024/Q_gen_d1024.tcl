add_design "Q_gen_d1024.v"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module Q_gen_d1024
synthesize -ads -selected_syn_tool_opt 2
