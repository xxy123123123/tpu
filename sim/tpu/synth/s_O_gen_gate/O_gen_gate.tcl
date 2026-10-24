add_design "O_gen_gate.v"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module O_gen_gate
synthesize -ads -selected_syn_tool_opt 2
