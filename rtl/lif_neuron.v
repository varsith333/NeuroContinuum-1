// =============================================================================
// lif_neuron.v  —  Leaky Integrate-and-Fire neuron (fixed point)
//
// The atomic building block of NeuroContinuum-1.  Behavioural reference:
//   simulator/neurons.py :: LIFLayer
//
// Membrane (forward Euler, one update per `en` pulse, which is one "timestep"):
//     v  <-  v + i_ext - (v >>> LEAK_SHIFT)
//     if v >= THRESHOLD : spike, v <- RESET_VAL, enter refractory for REFRACTORY
//
// Number format: signed two's-complement, DATA_W bits (default Q8.8).
//   LEAK_SHIFT implements the membrane time-constant: tau ~ 2**LEAK_SHIFT steps.
//   A larger LEAK_SHIFT => slower leak => longer memory, lower firing rate.
//
// This is a plain (deterministic) neuron.  stochastic_neuron.v wraps this idea
// with a programmable noise injection for the uncertainty feature.
// =============================================================================
`timescale 1ns/1ps

module lif_neuron #(
    parameter integer DATA_W      = 16,   // membrane / current width (signed)
    parameter integer LEAK_SHIFT  = 5,    // v leaks by v>>LEAK_SHIFT each step
    parameter signed [DATA_W-1:0] THRESHOLD = 16'sd400, // spike if v >= THRESHOLD
    parameter signed [DATA_W-1:0] RESET_VAL = 16'sd0,   // v after a spike
    parameter integer REFRACTORY  = 3     // steps of absolute refractory
) (
    input  wire                   clk,
    input  wire                   rst,     // sync reset (active high)
    input  wire                   en,      // advance one timestep when high
    input  wire signed [DATA_W-1:0] i_ext, // input synaptic current
    output reg                    spike,   // 1-clk pulse on spike
    output wire signed [DATA_W-1:0] v_out  // live membrane potential (probe)
);

    reg signed [DATA_W-1:0] v;
    reg [7:0]               ref_cnt;

    // leaky term (arithmetic shift keeps the sign)
    wire signed [DATA_W-1:0] leak    = v >>> LEAK_SHIFT;
    // candidate next membrane = integrate current, remove leak
    wire signed [DATA_W-1:0] v_next  = v + i_ext - leak;
    wire                     fire    = (v_next >= THRESHOLD) && (ref_cnt == 0);

    always @(posedge clk) begin
        if (rst) begin
            v       <= RESET_VAL;
            spike   <= 1'b0;
            ref_cnt <= 8'd0;
        end else begin
            spike <= 1'b0;                       // default: single-cycle pulse
            if (en) begin
                if (ref_cnt != 0) begin
                    ref_cnt <= ref_cnt - 8'd1;   // still refractory: hold at reset
                    v       <= RESET_VAL;
                end else begin
                    v <= fire ? RESET_VAL : v_next;
                    if (fire) begin
                        spike   <= 1'b1;
                        ref_cnt <= REFRACTORY;
                    end
                end
            end
        end
    end

    assign v_out = v;

endmodule
