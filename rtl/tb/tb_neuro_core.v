// tb_neuro_core.v — self-checking testbench for the integrated core.
//
// Checks, in order:
//   1. the output stage emits spikes and the first-spike decoder latches a
//      winner (NOVEL #3 + neuron sanity),
//   2. injecting a STUCK synapse cell (NOVEL #4) is detected via write
//      refusal and the spare-column remap actually repairs it,
//   3. the core keeps running after the repair (spikes observed post-repair).
//
// Stimulus is driven on the NEGEDGE so it is stable well before the DUT's
// posedge sampling — no simulator-order races.  Input spikes are held for the
// whole 2-clock timestep, which is the core's registered input interface.
//
// Run:  make sim_core      (or `make wave` for a VCD: build/wave.vcd)
`timescale 1ns/1ps
module tb_neuro_core;
    localparam N_IN = 4, N_HID = 4, N_OUT = 4, ID_W = 2;

    reg clk = 0, rst = 1, step = 0, noise_on = 1, learn_en = 1;
    reg [N_IN-1:0] in_spikes = 0;
    reg [N_HID-1:0] inject_stuck = 0;
    reg reward_valid = 0;
    reg signed [7:0] reward = 0;
    reg [ID_W-1:0] label = 0;

    wire [N_OUT-1:0] out_spikes;
    wire [ID_W-1:0]  winner;
    wire             winner_valid, fault_any, repaired_any, degraded_any;

    neuro_core #(.N_IN(N_IN), .N_HID(N_HID), .N_OUT(N_OUT), .ID_W(ID_W)) dut (
        .clk(clk), .rst(rst), .step(step), .in_spikes(in_spikes),
        .noise_on(noise_on), .learn_en(learn_en),
        .inject_stuck(inject_stuck),
        .reward_valid(reward_valid), .reward(reward), .label(label),
        .out_spikes(out_spikes), .winner(winner),
        .winner_valid(winner_valid), .fault_any(fault_any),
        .repaired_any(repaired_any), .degraded_any(degraded_any));

    always #5 clk = ~clk;

    `ifdef DUMP
        initial begin
            $dumpfile("wave.vcd");
            $dumpvars(0, tb_neuro_core);
        end
    `endif

    integer out_count = 0, won = 0, repaired = 0, faults = 0, spikes_after = 0;
    integer t;

    // observers (count 1-clk spike pulses wherever they land)
    always @(posedge clk) if (!rst) begin
        if (|out_spikes)  out_count = out_count + 1;
        if (winner_valid) won = 1;
        if (repaired_any) repaired = repaired + 1;
        if (fault_any)    faults = faults + 1;
        if (repaired > 0 && |out_spikes) spikes_after = spikes_after + 1;
    end

    // one input pattern per timestep (t), valid for the whole timestep
    task drive_step;
        input integer tt;
        begin
            in_spikes = (tt % 10 == 0) ? 4'b0011 :
                        (tt % 10 == 5) ? 4'b1100 : 4'b0000;
            step = 1;
            @(negedge clk);   // posedge sampled: step=1, spikes in
            step = 0;
            @(negedge clk);   // posedge sampled: step=0, spikes still held
                              // (this edge is where the 1-clk hidden spike
                              //  pulse overlaps the input -> coincidence)
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst = 0;

        // ---- phase 1: healthy run, first-spike decision --------------------
        for (t = 0; t < 60; t = t + 1) drive_step(t);

        // ---- phase 2: poison row 0's live weight cell (fault injection) -----
        inject_stuck = 4'b0001;      // seen by exactly one posedge
        @(negedge clk);
        inject_stuck = 4'b0000;

        // continue driving; detection + remap should complete well within this
        for (t = 0; t < 300; t = t + 1) drive_step(t);

        // ---- checks ---------------------------------------------------------
        $display("[CORE] output spikes observed : %0d %s",
                 out_count, (out_count > 0) ? "OK" : "FAIL");
        $display("[CORE] first-spike winner      : %0d (latched=%0d) %s",
                 winner, won, won ? "OK" : "FAIL");
        $display("[CORE] stuck-synapse detections: %0d", faults);
        $display("[CORE] spare-column repairs    : %0d %s",
                 repaired, (repaired > 0) ? "OK" : "FAIL");
        $display("[CORE] output spikes after fix : %0d %s",
                 spikes_after, (spikes_after > 0) ? "OK" : "FAIL");
        if (out_count == 0) $fatal(1, "core produced no output spikes");
        if (!won)           $fatal(1, "decoder never latched a winner");
        if (faults == 0)    $fatal(1, "injected stuck cell was never detected");
        if (repaired == 0)  $fatal(1, "spare-column remap never fired");
        if (spikes_after == 0) $fatal(1, "core stalled after the repair");
        $display("TB_CORE_PASS");
        $finish;
    end
endmodule
