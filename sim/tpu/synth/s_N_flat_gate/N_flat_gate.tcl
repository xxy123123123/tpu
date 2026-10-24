add_design "N_flat_gate.v"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module N_flat_gate
synthesize -ads -selected_syn_tool_opt 2
