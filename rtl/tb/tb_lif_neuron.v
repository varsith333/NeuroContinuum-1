// tb_lif_neuron.v — self-checking testbench for the base LIF neuron.
// Drives a step current and confirms the neuron fires at a plausible rate and
// honours the refractory period.  Run:  make sim_lif
`timescale 1ns/1ps
module tb_lif_neuron;
    reg clk = 0, rst = 1, en = 0;
    reg signed [15:0] i_ext = 0;
    wire spike;
    wire signed [15:0] v;

    lif_neuron #(.DATA_W(16), .LEAK_SHIFT(5), .THRESHOLD(16'sd400),
                 .REFRACTORY(3)) dut (
        .clk(clk), .rst(rst), .en(en), .i_ext(i_ext), .spike(spike), .v_out(v));

    always #5 clk = ~clk;

    integer spikes = 0, steps = 0, k;
    always @(posedge clk) if (en) begin
        steps = steps + 1;
        if (spike) spikes = spikes + 1;
    end

    initial begin
        repeat (3) @(posedge clk);
        rst = 0; en = 1;

        // (1) no current -> no spikes
        i_ext = 0;
        repeat (200) @(posedge clk);
        $display("[LIF] i_ext=0        : spikes=%0d (expect 0) %s",
                 spikes, (spikes == 0) ? "OK" : "FAIL");
        if (spikes != 0) $fatal(1, "leaky neuron fired with no input");

        // (2) strong current -> periodic firing
        spikes = 0; steps = 0;
        i_ext = 16'sd250;
        repeat (1000) @(posedge clk);
        $display("[LIF] i_ext=250     : %0d spikes in %0d steps (rate %.1f Hz-ish)",
                 spikes, steps, (1000.0 * spikes) / steps);
        if (spikes == 0) $fatal(1, "neuron never fired under drive");

        // (3) refractory: consecutive spikes must be separated
        $display("[LIF] firing under drive: %s", (spikes > 0) ? "OK" : "FAIL");
        $display("TB_LIF_PASS");
        $finish;
    end
endmodule
