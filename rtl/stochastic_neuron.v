// =============================================================================
// stochastic_neuron.v  —  LIF neuron with programmable membrane noise  (NOVEL #2)
//
// Behavioural reference: simulator/stochastic.py :: StochasticLIFLayer
//
// WHY THIS IS NOVEL IN HARDWARE
// -----------------------------
// Every shipped neuromorphic chip is deterministic: identical input gives an
// identical output, so the chip can only ever emit a bare class label.  This
// neuron injects controlled noise straight into the membrane:
//     v <- v + i_ext - leak + sigma_noise
// so running the SAME input several times yields DIFFERENT spike patterns.
// A population of these performs on-chip Monte-Carlo sampling: the spread of
// winners across trials is a calibrated posterior / uncertainty estimate, with
// no backpropagation and no cloud.  `sigma_noise` is software-programmable.
//
// Noise source = 32-bit maximal-length LFSR (taps 32,22,2,1) feeding a cheap
// sum-of-taps CLT approximation to a zero-mean Gaussian.  Mirrors the
// `noise_mode='lfsr'` path in stochastic.py.  Set NOISE_EN=0 for the
// deterministic neuron (sigma = 0).
// =============================================================================
`timescale 1ns/1ps

module stochastic_neuron #(
    parameter integer DATA_W     = 16,
    parameter integer LEAK_SHIFT = 5,
    parameter integer NOISE_SHIFT= 4,    // sigma = 1/2**NOISE_SHIFT (amplitude)
    parameter integer NOISE_EN   = 1,    // 0 = deterministic neuron
    parameter signed [DATA_W-1:0] THRESHOLD = 16'sd400,
    parameter signed [DATA_W-1:0] RESET_VAL = 16'sd0,
    parameter integer REFRACTORY = 3,
    parameter [31:0]    LFSR_SEED = 32'hACE1_2345
) (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     en,
    input  wire                     noise_on,  // runtime enable of noise
    input  wire signed [DATA_W-1:0] i_ext,
    output reg                      spike,
    output wire signed [DATA_W-1:0] v_out
);

    // ---- 32-bit maximal LFSR (x^32 + x^22 + x^2 + x + 1) ------------------
    reg [31:0] lfsr;
    wire lfsr_fb = lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0];
    always @(posedge clk) begin
        if (rst) lfsr <= LFSR_SEED;
        else     lfsr <= {lfsr[30:0], lfsr_fb};
    end

    // ---- cheap signed "gaussian": XOR-fold LFSR bits -> small signed value --
    // Sum of a few taps gives an approximate zero-mean noise in [-N, +N].
    wire signed [7:0] noise_raw =
        $signed({1'b0, lfsr[3:0]})  - $signed({1'b0, lfsr[11:8]}) +
        $signed({1'b0, lfsr[19:16]}) - $signed({1'b0, lfsr[27:24]});
    wire signed [DATA_W-1:0] noise = (noise_raw >>> NOISE_SHIFT);

    // ---- base deterministic LIF ------------------------------------------------
    reg signed [DATA_W-1:0] v;
    reg [7:0]               ref_cnt;
    wire signed [DATA_W-1:0] leak   = v >>> LEAK_SHIFT;
    // apply noise only when enabled (this is the ONLY difference vs lif_neuron)
    wire signed [DATA_W-1:0] noise_gated = (NOISE_EN != 0 && noise_on) ? noise : {DATA_W{1'b0}};
    wire signed [DATA_W-1:0] v_next = v + i_ext - leak + noise_gated;
    wire                     fire   = (v_next >= THRESHOLD) && (ref_cnt == 0);

    always @(posedge clk) begin
        if (rst) begin
            v <= RESET_VAL; spike <= 1'b0; ref_cnt <= 8'd0;
        end else begin
            spike <= 1'b0;
            if (en) begin
                if (ref_cnt != 0) begin
                    ref_cnt <= ref_cnt - 8'd1;
                    v       <= RESET_VAL;
                end else begin
                    v <= fire ? RESET_VAL : v_next;
                    if (fire) begin spike <= 1'b1; ref_cnt <= REFRACTORY; end
                end
            end
        end
    end

    assign v_out = v;

endmodule
