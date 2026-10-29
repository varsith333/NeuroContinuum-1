// =============================================================================
// neuro_core.v  —  NeuroContinuum-1 processing core (integrates all 4 novelties)
//
// Datapath (one core):
//
//   input spikes (N_IN)
//        │  weight from the physical synapse array (BCM, self-healing cells)
//        ▼
//   hidden stochastic neurons (N_HID)   <- NOVEL #1 BCM   (bcm_plasticity)
//                                         NOVEL #2 noise  (stochastic_neuron)
//                                         NOVEL #4 repair (fault_monitor +
//                                                         synapse_remap)
//        │
//        ▼
//   output racers (N_OUT)               <- NOVEL #3 first_spike_decoder
//
// This is the synthesizable structural top for a single core.  It is a small
// (4-in / 4-hidden / 4-out) instantiation chosen to keep simulation fast while
// exercising every block and its wiring; scale N_* via parameters.
//
// HOW THE FOUR NOVELTIES ARE WIRED (this is real wiring, not a stub)
// ------------------------------------------------------------------
// * Synapse array: `w_phys[row][col]` holds the physical weight cells.
//   `live_col[row]` is the logical -> physical column map.  At reset each row
//   sits on its own input column (identity map).
// * NOVEL #1: bcm_plasticity reads `w_cur` from the LIVE cell and computes
//   `w_next`/`move`; the row writes `w_next` back — unless the cell is stuck.
// * NOVEL #4: a stuck cell REFUSES the write, so `w_req` (registered request)
//   stops matching `w_phys` (actual).  fault_monitor counts those refusals and
//   pulses a one-shot `fault`; arbitration picks a pending row and the shared
//   synapse_remap hands back a spare column; the row re-points `live_col`.
//   `inject_stuck` is a test/DFT port that poisons a row's live cell so a
//   testbench can prove the whole detection -> repair loop end to end.
//   A row that runs out of spares latches `degraded` (graceful degradation).
//
// COMPACT-CORE SIMPLIFICATION (stated honestly, matches the Python model):
//   each hidden row has ONE synapse (row -> input row%N_IN).  A repaired spare
//   column mirrors that SAME input, so no input mux is needed here; a full
//   crossbar would add a per-row input mux driven by the spare-source table.
//
// The reward-gated readout learning (continuum_net.py::_reward_readout) is
// applied to the output stage by the host through the `reward`+`label` ports
// (a global teaching line); this core only forms output spikes + a winner.
// =============================================================================
`timescale 1ns/1ps

module neuro_core #(
    parameter integer N_IN    = 4,
    parameter integer N_HID   = 4,
    parameter integer N_OUT   = 4,
    parameter integer DATA_W  = 16,
    parameter integer W_W     = 8,
    parameter integer N_SPARE = 2,
    parameter integer ID_W    = 2,
    parameter integer STUCK_CYCLES = 3,
    parameter signed [W_W-1:0] W_INIT = 8'sd40   // reset value of every cell
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 step,        // advance one timestep
    input  wire [N_IN-1:0]      in_spikes,   // input layer spikes
    input  wire                 noise_on,    // enable stochastic neurons
    input  wire                 learn_en,    // enable BCM plasticity
    // ---- fault injection (test / DFT port) --------------------------------
    input  wire [N_HID-1:0]     inject_stuck, // poison each row's live cell
    // ---- host teaching interface (reward-gated readout learning) ----------
    input  wire                 reward_valid,
    input  wire signed [7:0]    reward,      // +/- scalar reward
    input  wire [ID_W-1:0]      label,       // correct class (teacher route)
    // ---- outputs ---------------------------------------------------------
    output wire [N_OUT-1:0]     out_spikes,  // this-step output spikes
    output wire [ID_W-1:0]      winner,      // first-spike decision
    output wire                 winner_valid,
    output wire                 fault_any,    // a stuck cell was DETECTED
    output wire                 repaired_any, // remap re-pointed a synapse
    output wire                 degraded_any  // ran out of spares (graceful)
);

    localparam integer NC    = N_IN + N_SPARE;   // physical columns per row
    localparam integer COL_W = (NC <= 2) ? 1 : $clog2(NC);

    // ---- synapse array state (one BCM synapse per row in this compact core) -
    reg [W_W-1:0]        w_phys   [0:N_HID-1][0:NC-1]; // physical weight cells
    reg [COL_W-1:0]      live_col [0:N_HID-1];          // logical->physical col
    reg [NC-1:0]         stuck    [0:N_HID-1];          // per-cell stuck mask
    reg signed [W_W-1:0] w_req    [0:N_HID-1];          // last write request
    reg                  move_q   [0:N_HID-1];          // rule wanted a change
    reg [N_HID-1:0]      pending;                       // repair not yet done
    reg [N_HID-1:0]      degraded;                      // row lost its spare

    wire [N_HID-1:0]     fault_row;   // per-row one-shot fault pulses
    wire [N_HID-1:0]     repair_now;  // remap finished for this row, this clk
    wire [N_HID-1:0]     hid_spikes;

    // remap handshake (declared before first use in the row generate)
    wire              map_done;
    wire              map_ok;
    wire [ID_W-1:0]   map_id;
    wire [7:0]        map_out;
    wire [COL_W-1:0]  map_out_q = map_out[COL_W-1:0];
    reg               servicing;   // one repair request in flight
    reg [ID_W-1:0]    sel_id;      // arbitration: chosen faulting row (index)
    reg               sel_valid;   // ...and that some row is waiting for a repair
    wire              remap_req = sel_valid && !servicing && !map_done;

    // =========================================================================
    // hidden rows: weight -> BCM -> stochastic neuron -> fault detection
    // =========================================================================
    genvar gi;
    generate
        for (gi = 0; gi < N_HID; gi = gi + 1) begin : g_row
            localparam integer SRC_I = gi % N_IN;   // input this row listens to

            integer c;
            reg signed [W_W-1:0]    w_cur;    // value in the live cell
            wire signed [W_W-1:0]   w_next;   // BCM desired value
            wire                    move;     // BCM requests a change
            wire                    wr_en;    // this cycle is a write attempt

            // read the live physical cell
            always @* w_cur = w_phys[gi][live_col[gi]];

            // a write (and a detection sample) happens on a pre/post
            // COINCIDENCE: the BCM rule only moves weights on coincidence, so
            // write-refusals can only be observed at those instants.
            // (input spikes are held for the whole timestep, so the 1-clk post
            //  pulse at t+1 still overlaps the input window -> pre && post fire)
            assign wr_en = learn_en && in_spikes[SRC_I] && hid_spikes[gi];

            // ---- NOVEL #1: BCM sliding-threshold meta-plasticity -----------
            // activity/threshold traces run continuously at clk rate
            bcm_plasticity #(.W_W(W_W)) u_bcm (
                .clk(clk), .rst(rst), .en(learn_en),
                .pre(in_spikes[SRC_I]), .post(hid_spikes[gi]),
                .w_cur(w_cur), .w_next(w_next), .move(move), .theta()
            );

            // ---- NOVEL #2: stochastic (noise-driven) neuron -----------------
            // drive = live weight sign-extended and scaled into current units.
            // Gain x4 so a couple of coincident input spikes clear the 300
            // threshold with margin (v asymptote must exceed THRESHOLD).
            wire signed [DATA_W-1:0] w_scaled =
                {{(DATA_W-W_W){w_cur[W_W-1]}}, w_cur} <<< 2;
            wire signed [DATA_W-1:0] i_ext =
                in_spikes[SRC_I] ? w_scaled : {DATA_W{1'b0}};

            stochastic_neuron #(
                .DATA_W(DATA_W), .LEAK_SHIFT(5), .NOISE_SHIFT(4), .NOISE_EN(1),
                .THRESHOLD(16'sd300), .REFRACTORY(2)
            ) u_neu (
                .clk(clk), .rst(rst), .en(step),
                .noise_on(noise_on), .i_ext(i_ext),
                .spike(hid_spikes[gi]), .v_out()
            );

            // ---- NOVEL #4a: stuck-cell detector (write refusal) -------------
            fault_monitor #(.W_W(W_W), .STUCK_CYCLES(STUCK_CYCLES)) u_mon (
                .clk(clk), .rst(rst), .en(wr_en),
                .w_want(w_req[gi]), .w_actual(w_cur),
                .rule_wanted_change(move_q[gi]),
                .fault(fault_row[gi]), .stall_count()
            );

            // ---- weight cell write / request bookkeeping --------------------
            always @(posedge clk) begin
                if (rst) begin
                    stuck[gi]  <= {NC{1'b0}};
                    w_req[gi]  <= {W_W{1'b0}};
                    move_q[gi] <= 1'b0;
                    for (c = 0; c < NC; c = c + 1)
                        w_phys[gi][c] <= W_INIT;
                end else if (repair_now[gi]) begin
                    // repair landed: void the old request, adopt the new cell
                    w_req[gi]  <= w_phys[gi][map_out_q[COL_W-1:0]];
                    move_q[gi] <= 1'b0;
                end else if (wr_en) begin
                    w_req[gi]  <= w_next;
                    move_q[gi] <= move;
                    // a STUCK cell refuses this write (the detectable event)
                    if (!stuck[gi][live_col[gi]])
                        w_phys[gi][live_col[gi]] <= w_next;
                end
                // test/DFT: poison the currently-live cell of this row
                if (inject_stuck[gi])
                    stuck[gi][live_col[gi]] <= 1'b1;
            end
        end
    endgenerate

    // =========================================================================
    // NOVEL #4b: repair arbitration + shared spare-column allocator
    // =========================================================================
    integer pi;
    always @* begin
        sel_valid = 1'b0;
        sel_id    = {ID_W{1'b0}};
        for (pi = N_HID - 1; pi >= 0; pi = pi - 1)
            if (pending[pi]) begin
                sel_valid = 1'b1;
                sel_id    = pi;              // low index wins
            end
    end

    // the input this synapse listens to (spare columns mirror this input)
    wire [31:0] sel_src = sel_id % N_IN;
    // the column the row currently sits on (never re-pick a failed cell)
    wire [7:0]  sel_cur = live_col[sel_id];

    // (remap handshake wires are declared before the row generate above)

    synapse_remap #(.N_COL(N_IN), .S(N_SPARE), .ID_W(ID_W)) u_remap (
        .clk(clk), .rst(rst),
        .fault(remap_req), .fault_id(sel_id),
        .fault_src(sel_src), .cur_col(sel_cur),
        .done_valid(map_done), .done_ok(map_ok), .done_id(map_id),
        .map_out(map_out), .spares_left()
    );

    // announce which rows are being repaired this cycle (drives w_req/move_q)
    genvar gj;
    generate
        for (gj = 0; gj < N_HID; gj = gj + 1) begin : g_repair_now
            assign repair_now[gj] = map_done && map_ok && (map_id == gj);
        end
    endgenerate

    integer ri;
    always @(posedge clk) begin
        if (rst) begin
            for (ri = 0; ri < N_HID; ri = ri + 1) begin
                live_col[ri] <= ri % N_IN;   // identity map at reset
                pending[ri]  <= 1'b0;
                degraded[ri] <= 1'b0;
            end
            servicing <= 1'b0;
        end else begin
            servicing <= sel_valid && !servicing && !map_done;
            // latch new fault requests (a degraded row stops requesting)
            for (ri = 0; ri < N_HID; ri = ri + 1)
                if (fault_row[ri] && !degraded[ri])
                    pending[ri] <= 1'b1;
            // consume the allocator's answer
            if (map_done) begin
                pending[map_id] <= 1'b0;
                if (map_ok)
                    live_col[map_id] <= map_out_q;   // re-point to the spare
                else
                    degraded[map_id] <= 1'b1;        // graceful degradation
            end
        end
    end

    assign fault_any    = |fault_row;
    assign repaired_any = map_done && map_ok;
    assign degraded_any = map_done && !map_ok;

    // =========================================================================
    // NOVEL #3: first-spike temporal decoder over the output racers
    // =========================================================================
    first_spike_decoder #(.N(N_OUT), .ID_W(ID_W)) u_dec (
        .clk(clk), .rst(rst), .clr(reward_valid),  // re-arm after each reward
        .spikes(out_spikes), .tick(step),
        .winner(winner), .winner_valid(winner_valid), .race_t()
    );

    // compact output stage: drive output neurons from hidden.
    // A neuron's 1-clk spike pulse lands between `step` edges, so it is HELD in
    // hid_evt and consumed exactly once by the next enabled step (an AER-style
    // event buffer — without it the layer loses every inter-layer spike).
    reg [N_HID-1:0] hid_evt;
    always @(posedge clk)
        if (rst)                   hid_evt <= {N_HID{1'b0}};
        else if (step)             hid_evt <= hid_spikes;      // consume window
        else                       hid_evt <= hid_evt | hid_spikes;

    wire [N_OUT-1:0] out_pre;
    generate
        for (gi = 0; gi < N_OUT; gi = gi + 1) begin : g_out
            stochastic_neuron #(
                .DATA_W(DATA_W), .LEAK_SHIFT(4), .NOISE_SHIFT(5), .NOISE_EN(1),
                .THRESHOLD(16'sd150), .REFRACTORY(1)
            ) u_out (
                .clk(clk), .rst(rst), .en(step),
                .noise_on(noise_on),
                .i_ext(hid_evt[gi % N_HID] ? 16'sd160 : 16'sd0),
                .spike(out_pre[gi]), .v_out()
            );
        end
    endgenerate
    assign out_spikes = out_pre;

endmodule
