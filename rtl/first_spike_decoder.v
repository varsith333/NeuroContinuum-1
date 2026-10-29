// =============================================================================
// first_spike_decoder.v  —  Race-based winner-take-first readout  (NOVEL #3)
//
// Behavioural reference: simulator/temporal.py :: FirstSpikeDecoder
//
// WHY THIS IS NOVEL IN HARDWARE
// -----------------------------
// Rate coding (what shipped chips use) needs tens of spikes to convey one
// number and a accumulator/multiplier to read it out.  Temporal coding encodes
// intensity in WHEN a neuron fires: stronger input -> earlier spike.  This
// decoder just RACES the input neurons; the first one to spike latches as the
// winner and its identity is the decision.  No accumulator, no multiply — just
// a priority-encode + latch + timer.  Sub-millisecond, ultra-sparse inference.
//
// Operation:
//   * `spikes[N-1:0]` is the one-hot-per-timestep race input (combinational).
//   * Before a winner is latched, the lowest-index spiking neuron wins (a
//     simultaneous tie resolves to the lowest id, deterministic).
//   * `winner_valid` rises for one clock when `winner` is captured and `race_t`
//     holds the timestamp of the winning spike (decision latency in steps).
//   * `clr` re-arms the decoder for the next race.
// =============================================================================
`timescale 1ns/1ps

module first_spike_decoder #(
    parameter integer N        = 4,    // number of racing neurons / classes
    parameter integer ID_W     = 2,    // ceil(log2(N))
    parameter integer T_W      = 16    // race timestamp width
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              clr,       // re-arm for a new race
    input  wire [N-1:0]      spikes,    // this-step spikes (bit per neuron)
    input  wire              tick,      // advance the race timer
    output reg  [ID_W-1:0]   winner,    // latched winner id
    output reg               winner_valid, // 1-clk pulse when winner captured
    output reg  [T_W-1:0]    race_t     // timestamp of the winning spike
);

    reg             armed;   // still looking for a spike
    reg [T_W-1:0]   t;

    // priority encode the lowest set bit (lowest id wins a same-cycle tie)
    integer i;
    reg [ID_W-1:0] enc_id;
    reg            enc_hit;
    always @* begin
        enc_id   = {ID_W{1'b0}};
        enc_hit  = 1'b0;
        for (i = N - 1; i >= 0; i = i - 1) begin
            if (spikes[i]) begin
                enc_id  = i[ID_W-1:0];
                enc_hit = 1'b1;
            end
        end
    end

    always @(posedge clk) begin
        if (rst || clr) begin
            armed        <= 1'b1;
            t            <= {T_W{1'b0}};
            winner       <= {ID_W{1'b0}};
            winner_valid <= 1'b0;
            race_t       <= {T_W{1'b0}};
        end else begin
            winner_valid <= 1'b0;
            if (tick && armed)
                t <= t + 1'b1;
            // first spike latches the winner and freezes the race
            if (armed && enc_hit) begin
                winner       <= enc_id;
                race_t       <= t;
                winner_valid <= 1'b1;
                armed        <= 1'b0;
            end
        end
    end

endmodule
