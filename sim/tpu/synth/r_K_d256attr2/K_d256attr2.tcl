add_design "K_d256attr2.v"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module K_d256attr2
synthesize -ads -selected_syn_tool_opt 2
