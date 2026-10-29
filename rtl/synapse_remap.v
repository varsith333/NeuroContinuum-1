// =============================================================================
// synapse_remap.v  —  spare-column allocator / remap table  (NOVEL #4 part 2)
//
// Behavioural reference: simulator/selfhealing.py :: SelfHealingSynapseArray
//
// When a fault is presented for logical synapse `fault_id` (whose input is
// `fault_src`), this unit repairs it by RE-POINTING that synapse at a spare
// physical column.  Spare columns are extra physical columns not connected to
// any input at reset; a claimed spare becomes a MIRROR of the faulty synapse's
// input (`spare_owner[s] = fault_src`), so the network sees the same input
// width either way — repairs are transparent.  A spare may be re-shared only if
// it already mirrors the same input (one wire, one signal — the constraint the
// Python model enforces).  If no spare qualifies, `done_ok` returns low
// (graceful degradation instead of a crash).
//
// Handshake: present `fault`+`fault_id`+`fault_src` for ONE cycle.  One cycle
// later the result is on `done_valid`/`done_ok`/`done_id`/`map_out`
// (`done_ok=1` -> `map_out` holds the new physical column).
//
// One repair per cycle; the caller serialises requests (neuro_core.v).
// =============================================================================
`timescale 1ns/1ps

module synapse_remap #(
    parameter integer N_COL = 4,   // live (logical) input columns
    parameter integer S     = 2,   // spare physical columns
    parameter integer ID_W  = 2    // bits for fault_id (logical synapse id)
) (
    input  wire               clk,
    input  wire               rst,
    input  wire               fault,        // repair request (present 1 cycle)
    input  wire [ID_W-1:0]    fault_id,     // logical synapse that failed
    input  wire [31:0]        fault_src,    // input that synapse listens to
    input  wire [7:0]         cur_col,      // column it currently sits on
                                            // (never re-pick the failed cell)
    output reg                done_valid,   // 1-clk result pulse
    output reg                done_ok,      // 1 = repaired (map_out valid)
    output reg  [ID_W-1:0]    done_id,      // which synapse the result is for
    output reg  [7:0]         map_out,      // new physical column (if ok)
    output reg  [7:0]         spares_left
);

    // spare_owner[s]: FREE, or the input index this spare mirrors (33 bits so
    // FREE can never collide with a real src value, which fits in 32 bits)
    localparam [32:0] FREE = 33'b1_1111_1111_1111_1111_1111_1111_1111_1111;

    reg [32:0] spare_owner [0:S-1];
    integer k;
    integer choice;

    always @(posedge clk) begin
        done_valid <= 1'b0;
        if (rst) begin
            for (k = 0; k < S; k = k + 1)
                spare_owner[k] <= FREE;
            spares_left <= S;
            done_ok     <= 1'b0;
            done_id     <= {ID_W{1'b0}};
            map_out     <= 8'd0;
        end else if (fault) begin
            // choose a spare that (a) is not the row's current (failed) cell,
            // (b) is free, or (c) already mirrors this input
            choice = -1;
            for (k = 0; k < S; k = k + 1) begin
                if (choice == -1 &&
                    ((N_COL + k) != cur_col) &&
                    (spare_owner[k] == FREE || spare_owner[k] == fault_src))
                    choice = k;
            end
            done_valid <= 1'b1;
            done_id    <= fault_id;
            if (choice != -1) begin
                map_out <= N_COL + choice[7:0];
                if (spare_owner[choice] == FREE) begin
                    spare_owner[choice] <= fault_src;   // spare mirrors input
                    spares_left         <= spares_left - 8'd1;
                end
                done_ok <= 1'b1;
            end else begin
                done_ok <= 1'b0;      // out of spares -> caller degrades the row
            end
        end
    end

    // parameter sanity (simulation only)
    initial begin
        if (S < 1) begin
            $display("synapse_remap: ERROR S must be >= 1");
            $finish;
        end
    end

endmodule
