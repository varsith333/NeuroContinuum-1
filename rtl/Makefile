# NeuroContinuum-1 RTL — simulation Makefile
#
# Primary flow uses Icarus Verilog (iverilog + vvp).  If you have Vivado on
# PATH instead, `make xsim` drives xvlog/xelab/xsim.  Both are self-checking:
# look for TB_*_PASS in the output.
#
#   make          # run everything
#   make sim_lif  # base neuron
#   make sim_core # integrated core
#   make wave     # core sim + dump a VCD you can open in GTKWave

IVERILOG ?= iverilog
VVP      ?= vvp
SRC      := lif_neuron.v bcm_plasticity.v stochastic_neuron.v \
            first_spike_decoder.v fault_monitor.v synapse_remap.v neuro_core.v
OUT      := work
$(shell mkdir -p $(OUT))

.PHONY: all sim_lif sim_core wave clean xsim

all: sim_lif sim_core

$(OUT)/lif.vvp: tb/tb_lif_neuron.v $(SRC)
	$(IVERILOG) -g2012 -o $@ tb/tb_lif_neuron.v $(SRC)

$(OUT)/core.vvp: tb/tb_neuro_core.v $(SRC)
	$(IVERILOG) -g2012 -o $@ tb/tb_neuro_core.v $(SRC)

sim_lif: $(OUT)/lif.vvp
	$(VVP) $< | tee $(OUT)/lif.log
	@grep -q TB_LIF_PASS $(OUT)/lif.log

sim_core: $(OUT)/core.vvp
	$(VVP) $< | tee $(OUT)/core.log
	@grep -q TB_CORE_PASS $(OUT)/core.log

wave: tb/tb_neuro_core.v $(SRC)
	$(IVERILOG) -g2012 -DDUMP -o $(OUT)/core.vvp tb/tb_neuro_core.v $(SRC)
	$(VVP) $(OUT)/core.vvp

# ---- Vivado xsim (second independent simulator) ----------------------------
# Windows without make: powershell -File rtl\run_xsim.ps1  (finds Vivado + both TBs)
# GUI project + waveforms:      vivado -mode batch -source rtl/create_project.tcl
xsim:
	xvlog $(SRC) tb/tb_neuro_core.v
	xelab tb_neuro_core -s core_snap
	xsim core_snap -R

clean:
	rm -rf $(OUT) *.vcd *.pb xsim.dir
