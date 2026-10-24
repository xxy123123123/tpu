add_design "G_modsync.v"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module G_modsync
synthesize -ads -selected_syn_tool_opt 2
