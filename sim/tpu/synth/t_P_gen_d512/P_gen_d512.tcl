add_design "P_gen_d512.v"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module P_gen_d512
synthesize -ads -selected_syn_tool_opt 2
