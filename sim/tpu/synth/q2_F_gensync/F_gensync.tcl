add_design "F_gensync.v"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module F_gensync
synthesize -ads -selected_syn_tool_opt 2
