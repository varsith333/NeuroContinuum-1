# report_util.tcl — synthesize neuro_core and print hardware-size numbers.
#
#   vivado -mode batch -source rtl\report_util.tcl
#
# Builds a throwaway project (so it never touches the GUI project, even if it
# is open), runs Vivado synthesis AND implementation on the structural core
# for the Artix-7 xc7a35tcpg236-1, and writes:
#
#   rtl\synth_run\synth_util.rpt      slice LUTs / flip-flops / BRAM / DSP
#   rtl\synth_run\synth_timing.rpt    worst-case timing vs a 100 MHz clock
#   rtl\synth_run\impl_util.rpt       the same, after place & route (definitive)
#   rtl\synth_run\impl_timing.rpt     post-route timing (definitive Fmax)
#   rtl\synth_run\impl_power.rpt      dynamic/static power estimate
#
# These are the FPGA numbers behind the area claims in
# docs/04_benchmarking.md. (FPGA LUT/FF are an *estimate* of ASIC area, not a
# tapeout — say so honestly anywhere they are quoted.)

set projDir [file normalize [file dirname [file normalize [info script]]]]
set runDir  [file join $projDir "synth_run"]

create_project nc_synth $runDir -part xc7a35tcpg236-1 -force
set_property target_language Verilog [current_project]

# one file per call, wrapped in [list ...]: paths contain spaces
foreach f [glob -nocomplain [file join $projDir "*.v"]] {
    add_files -fileset sources_1 [list $f]
}
set_property top neuro_core [get_filesets sources_1]
update_compile_order -fileset sources_1

# a fair timing target: 100 MHz (10 ns) on the testbench's clk port
set xdcFile [file join $runDir "timing.xdc"]
set fh [open $xdcFile w]
puts $fh "create_clock -period 10.000 -name sys_clk \[get_ports clk\]"
close $fh
add_files -fileset constrs_1 [list $xdcFile]

launch_runs synth_1 -jobs 4
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: synthesis did not complete"
    exit 1
}

open_run synth_1
report_utilization        -file [file join $runDir "synth_util.rpt"]
report_timing_summary     -file [file join $runDir "synth_timing.rpt"]

# headline numbers straight into the log
puts "=============================================================="
puts "SYNTHESIS COMPLETE  (neuro_core, xc7a35tcpg236-1, 100 MHz)"
puts "=============================================================="
report_utilization
puts "---------- worst-case timing (100 MHz target) ----------------"
report_timing_summary -delay_type min_max -max_paths 3 -warn_on_violation

# ---- place & route: the definitive numbers ----------------------------------
launch_runs impl_1 -jobs 4
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: implementation did not complete"
    exit 1
}

open_run impl_1
report_utilization    -file [file join $runDir "impl_util.rpt"]
report_timing_summary -file [file join $runDir "impl_timing.rpt"]
report_power          -file [file join $runDir "impl_power.rpt"]

puts "=============================================================="
puts "IMPLEMENTATION COMPLETE (placed & routed)"
puts "=============================================================="
report_utilization
puts "---------- worst-case timing, post-route --------------------"
report_timing_summary -delay_type min_max -max_paths 3 -warn_on_violation
puts "---------- dynamic power estimate ---------------------------"
report_power
puts "reports: [file join $runDir "synth_util.rpt"]"
puts "         [file join $runDir "synth_timing.rpt"]"
puts "         [file join $runDir "impl_util.rpt"]"
puts "         [file join $runDir "impl_timing.rpt"]"
puts "         [file join $runDir "impl_power.rpt"]"
