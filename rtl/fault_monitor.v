// =============================================================================
// fault_monitor.v  —  stuck-synapse detector via WRITE-REFUSAL  (NOVEL #4 part 1)
//
// Behavioural reference: simulator/selfhealing.py :: SelfHealingSynapseArray
//
// A healthy synapse always stores what its plasticity rule asked it to store.
// A STUCK cell (radiation upset, aged memristor pinned to a rail) rejects the
// write, so the stored value stops matching the request.  By watching for that
// MISMATCH — not merely "weight sitting at a rail" — we get zero false
// positives: a synapse that legitimately rests at w_max because the rule wants
// it there shows request == hold and is never flagged.
//
// Handshake (matches continuum_net.py::_screen):
//   * `w_want`        = the registered write request issued last `en`
//   * `w_actual`      = what the cell holds NOW (after that write attempt)
//   * `rule_wanted_change` = registered "the rule did request a different value"
// The mismatch must persist for STUCK_CYCLES consecutive confirmations (a real
// monitor debounces transient glitches) before `fault` is pulsed.  `fault` is
// a ONE-SHOT: firing clears the counter, so a confirmed fault requests exactly
// one repair (the remap unit never sees a repair storm).
// =============================================================================
`timescale 1ns/1ps

module fault_monitor #(
    parameter integer W_W          = 8,
    parameter integer STUCK_CYCLES = 3,
    parameter integer CNT_W        = 4
) (
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   en,                 // sample this step
    input  wire signed [W_W-1:0]  w_want,             // value the rule requested
    input  wire signed [W_W-1:0]  w_actual,           // value the cell holds
    input  wire                   rule_wanted_change, // rule intended a move
    output reg                    fault,              // 1-clk pulse: repair needed
    output wire [CNT_W-1:0]       stall_count
);

    reg [CNT_W-1:0] stall;
    wire refused = (w_want !== w_actual) && rule_wanted_change;
    wire [CNT_W-1:0] next_stall = stall + 1'b1;
    wire confirmed = refused && (next_stall >= STUCK_CYCLES[CNT_W-1:0]);

    always @(posedge clk) begin
        if (rst) begin
            stall <= {CNT_W{1'b0}};
            fault <= 1'b0;
        end else begin
            fault <= 1'b0;
            if (en) begin
                if (confirmed) begin
                    fault <= 1'b1;   // one-shot: one fault = one repair request
                    stall <= {CNT_W{1'b0}};
                end else begin
                    stall <= refused ? next_stall : {CNT_W{1'b0}};
                end
            end
        end
    end

    assign stall_count = stall;

endmodule
