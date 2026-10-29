# create_project.tcl — build and verify the Vivado GUI project for NeuroContinuum-1
#
# Run once from a shell (or the Vivado Tcl console):
#
#   vivado -mode batch -source rtl\create_project.tcl
#
# It creates rtl\NeuroContinuum\NeuroContinuum.xpr with:
#   * all RTL design sources (synthesis top = neuro_core)
#   * both self-checking testbenches as simulation sources
#     (simulation top = tb_neuro_core)
# and immediately runs the behavioral simulation once, so the project opens
# in the GUI already proven.  Afterwards just double-click the .xpr to open it.

set projDir  [file dirname [file normalize [info script]]]
set projName "NeuroContinuum"
set projFile [file join $projDir $projName "${projName}.xpr"]

# a common Artix-7 part; simulation does not care, and the part can be
# changed later in the GUI (Project Settings -> General)
create_project $projName [file join $projDir $projName] \
    -part xc7a35tcpg236-1 -force

set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# design sources (synthesis top: the structural core)
# NB: add one file per call, wrapped in [list ...] — paths contain spaces
# ("Default Project") and add_files re-splits a bare string at the space.
foreach f [glob -nocomplain [file join $projDir "*.v"]] {
    add_files -fileset sources_1 [list $f]
}
set_property top neuro_core [get_filesets sources_1]

# simulation sources (sim top: the self-checking core testbench)
foreach f [glob -nocomplain [file join $projDir "tb" "*.v"]] {
    add_files -fileset sim_1 [list $f]
}
set_property top tb_neuro_core [get_filesets sim_1]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

# ---- prove it: one behavioral simulation in batch mode -----------------------
# The testbench prints TB_CORE_PASS on success (it $fatal's on any failure).
launch_simulation
run all

puts "=============================================================="
puts "PROJECT READY: $projFile"
puts "Open it in the Vivado GUI, then: Flow Navigator ->"
puts "Run Simulation -> Run Behavioral Simulation (waveforms)."
puts "=============================================================="
