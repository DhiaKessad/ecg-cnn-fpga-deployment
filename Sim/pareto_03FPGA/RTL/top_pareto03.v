module top_pareto03(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  signed [15:0] data_in,
    output reg data_valid_out,
    output reg [15:0] data_out
);

    // --- Block 1: Conv(8) + BN + ReLU + Pool ---
    wire b1_valid;
    wire [127:0] b1_out; // 8 * 16
    block1_top u_b1 (
        .clk(clk), .reset(reset),
        .data_valid_in(data_valid_in), .data_in(data_in),
        .data_valid_out(b1_valid), .data_out(b1_out)
    );

    // --- Block 2: Conv(32) + BN + ReLU + Pool ---
    wire b2_valid;
    wire [511:0] b2_out; // 32 * 16
    block2_top u_b2 (
        .clk(clk), .reset(reset),
        .data_valid_in(b1_valid), .data_in(b1_out),
        .data_valid_out(b2_valid), .data_out(b2_out)
    );

    // --- Block 3: Conv(16) + BN + ReLU + Pool ---
    wire b3_valid;
    wire [255:0] b3_out; // 16 * 16
    block3_top u_b3 (
        .clk(clk), .reset(reset),
        .data_valid_in(b2_valid), .data_in(b2_out),
        .data_valid_out(b3_valid), .data_out(b3_out)
    );

    // --- Block 4: Conv(16) + BN + ReLU + Pool ---
    wire b4_valid;
    wire [255:0] b4_out; // 16 * 16
    block4_top u_b4 (
        .clk(clk), .reset(reset),
        .data_valid_in(b3_valid), .data_in(b3_out),
        .data_valid_out(b4_valid), .data_out(b4_out)
    );

    // --- Block 5: GAP + Dense0 + Dense1 + Argmax ---
    wire b5_valid;
    wire [15:0] b5_out; 
    block5_top u_b5 (
        .clk(clk), .reset(reset),
        .data_valid_in(b4_valid), .data_in(b4_out),
        .data_valid_out(b5_valid), .data_out(b5_out)
    );

    // --- Final Output Sync ---
    always @(posedge clk) begin
        if (reset) begin
            data_valid_out <= 0;
            data_out       <= 0;
        end else begin
            data_valid_out <= b5_valid;
            data_out       <= b5_out;
        end
    end

endmodule
