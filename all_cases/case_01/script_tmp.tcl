
open_project -reset project_tmp
set_top case_1
add_files case_1.cc
open_solution -reset "solution_tmp"
set_part {xc7z020clg484-1}
create_clock -period 10 -name default
source "./directive_tmp.tcl"
csynth_design
export_design -evaluate verilog -format ip_catalog
exit
