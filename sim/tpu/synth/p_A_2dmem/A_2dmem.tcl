add_design "A_2dmem.v"
set_arch -family Logos2 -device PG2L200H -speedgrade -6 -package FBB676
compile -top_module A_2dmem
