add_design "J_d256attr.v"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module J_d256attr
synthesize -ads -selected_syn_tool_opt 2
