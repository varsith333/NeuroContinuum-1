// =============================================================================
// bcm_plasticity.v  —  BCM meta-plasticity with a SLIDING threshold  (NOVEL #1)
//
// Behavioural reference: simulator/plasticity.py :: BCMPlasticity
//   dw     = eta * pre * post * (post - theta)      # LTP if post > theta
//   theta <- theta + eta_theta * (post^2 - theta)   # slow sliding threshold
//
// WHY THIS IS NOVEL IN HARDWARE
// -----------------------------
// Every shipped neuromorphic chip implements fixed-threshold Hebbian/STDP: the
// rule that moves a weight never changes its own criterion.  That ratchets
// weights to a rail or makes the network forget when input statistics drift.
// BCM judges every update against a threshold theta that ITSELF slides with a
// slow low-pass of recent activity squared, so one synapse flips between LTP
// and LTD and tracks a non-stationary world.  This is the continual-learning
// engine of the chip.
//
// Discrete (fixed point) form realised here, advancing every clock while
// `learn_en` is high:
//   // y = slow-ish leaky trace of postsynaptic spikes (the activity)
//   y      <- y - (y >>> ACC_SHIFT) + (post ? ACC_INC : 0)
//   // sliding threshold: slow low-pass of y^2   (y^2 = (y*y) >>> YSQ_SHIFT)
//   theta  <- theta + ((y_sq - theta) >>> TH_SHIFT)
//   // weight only moves on a pre/post COINCIDENCE (the Python pre*post gate):
//   y_now  = y + (post ? ACC_INC : 0)          # activity including this spike
//   diff   = y_now - theta                      # sign decides LTP vs LTD
//   w_next = sat(w_cur + ((diff + ROUND) >>> ETA_SHIFT))   # round-to-nearest
//   move   = (w_next != w_cur)                  # the rule REQUESTED a change
//
// Rounding (+ROUND before the shift) removes the fixed-point truncation bias
// that would otherwise push every small negative diff to -1 and none of the
// small positive diffs to +1.
//
// WEIGHT STORAGE IS EXTERNAL (interface note)
// -------------------------------------------
// `w_cur` is an INPUT and `w_next`/`move` are OUTPUTS: this module only
// COMPUTES the next weight value.  The physical weight cell lives in the synapse
// array (neuro_core.v) so that a STUCK cell can refuse the write — which is
// exactly what fault_monitor.v then detects (write-refusal, novel #4).
//
// `y` and `theta` both run at clk rate (they are analog-style filters, not
// timestep events); the weight write is gated by the coincidence elsewhere.
// =============================================================================
`timescale 1ns/1ps

module bcm_plasticity #(
    parameter integer W_W        = 8,     // weight width (signed)
    parameter integer ACC_W      = 12,    // activity / threshold width (signed)
    parameter integer ACC_SHIFT  = 2,     // y leak  (fast-ish: tau ~ 2**2 clks)
    parameter integer ACC_INC    = 32,    // y increment on a post spike
    parameter integer TH_SHIFT   = 4,     // theta follower gain (y^2 low-pass)
    parameter integer YSQ_SHIFT  = 5,     // y^2 scaling: y_sq = (y*y) >>> 5
    parameter integer THETA0     = 16,    // reset value of the sliding threshold
    parameter integer ETA_SHIFT  = 3,     // weight learning rate = 1/2**ETA_SHIFT
    parameter integer ROUND      = 4,     // half-LSB: (diff+ROUND) >>> ETA_SHIFT
    parameter signed [W_W-1:0] W_MAX =  8'sd120,
    parameter signed [W_W-1:0] W_MIN = -8'sd120
) (
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   en,        // advance y/theta traces
    input  wire                   pre,       // presynaptic spike
    input  wire                   post,      // postsynaptic spike
    input  wire signed [W_W-1:0]  w_cur,     // weight currently stored in the cell
    output wire signed [W_W-1:0]  w_next,    // desired value after this step
    output wire                   move,      // rule requests a change (w_next != w_cur)
    output wire signed [ACC_W-1:0] theta     // live sliding threshold (probe)
);

    localparam integer YW = 2 * ACC_W;      // width of y*y

    reg signed [ACC_W-1:0] y;                // filtered postsynaptic activity
    reg signed [ACC_W-1:0] th;               // sliding modification threshold

    // ---- y^2, scaled into the threshold's domain, saturated ---------------
    wire signed [YW-1:0] y_prod    = y * y;
    wire signed [YW-1:0] y_scaled  = y_prod >>> YSQ_SHIFT;
    localparam signed [YW-1:0] ACC_HI = (1 <<< (ACC_W-1)) - 1;
    wire signed [YW-1:0] y_clamped = (y_scaled > ACC_HI) ? ACC_HI : y_scaled;
    wire signed [ACC_W-1:0] y_sq    = y_clamped[ACC_W-1:0];

    // ---- sliding threshold: theta follows y^2 with a SLOW low-pass ---------
    wire signed [ACC_W-1:0] th_step = (y_sq - th) >>> TH_SHIFT;

    // ---- weight update (only on pre/post coincidence) ----------------------
    wire signed [ACC_W-1:0] y_now = y + (post ? ACC_INC : 0);
    wire signed [ACC_W-1:0] diff_raw = y_now - th;
    // clamp before adding ROUND so the sum cannot overflow ACC_W
    wire signed [ACC_W-1:0] diff = (diff_raw > 12'sd2000) ? 12'sd2000 : diff_raw;
    wire signed [ACC_W-1:0] dw   = (diff + ROUND) >>> ETA_SHIFT;   // round-to-nearest

    // Sign-extend the stored weight into the arithmetic domain, add, saturate.
    // All comparisons are SIGNED in the ACC_W domain (an earlier version used
    // unsigned concatenations, which would clamp every negative weight up).
    wire signed [ACC_W-1:0] w_ext     = {{(ACC_W-W_W){w_cur[W_W-1]}}, w_cur} + dw;
    wire signed [ACC_W-1:0] w_max_ext = {{(ACC_W-W_W){W_MAX[W_W-1]}}, W_MAX};
    wire signed [ACC_W-1:0] w_min_ext = {{(ACC_W-W_W){W_MIN[W_W-1]}}, W_MIN};
    wire signed [W_W-1:0]   w_sat     = (w_ext > w_max_ext) ? W_MAX :
                                        (w_ext < w_min_ext) ? W_MIN :
                                        w_ext[W_W-1:0];

    // the weight only moves on a coincidence (matches Python's pre*post gate)
    assign w_next = (pre && post) ? w_sat : w_cur;
    assign move   = (w_next != w_cur);

    always @(posedge clk) begin
        if (rst) begin
            y  <= {ACC_W{1'b0}};
            th <= THETA0;
        end else if (en) begin
            // 1. decay + increment the activity trace
            y  <= (y - (y >>> ACC_SHIFT)) + (post ? ACC_INC : 0);
            // 2. slide the threshold toward recent activity^2 (the novelty)
            th <= th + th_step;
        end
    end

    assign theta = th;

    // parameter sanity (simulation only; synthesis tools ignore initial blocks)
    initial begin
        if (ACC_W <= W_W) begin
            $display("bcm_plasticity: ERROR ACC_W(%0d) must exceed W_W(%0d)",
                     ACC_W, W_W);
            $finish;
        end
    end

endmodule
