module block5_top(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [255:0] data_in, // 16 channels * 16 bits = 256 bit output from Block 4
    output reg data_valid_out,
    output reg [15:0] data_out // Argmax classification choice [0, 1, 2, or 3]
);
    // --- GAP ---
    wire gap_valid;
    wire [255:0] gap_out;
    layer_gap #(
        .IN_CHANNELS(16),
        .TIME_STEPS(11) // From Block 4 shape [11, 16]
    ) u_gap (
        .clk(clk), 
        .reset(reset),
        .data_valid_in(data_valid_in), 
        .data_in(data_in),
        .data_valid_out(gap_valid), 
        .data_out(gap_out)
    );

    // --- Dense 0 ---
    wire d0_valid;
    wire [1023:0] d0_out;
    layer_dense0 #(
        .WEIGHT_FILE("hex/dense0_weights.hex"),
        .BIAS_FILE  ("hex/dense0_bias.hex")
    ) u_dense0 (
        .clk(clk), 
        .reset(reset),
        .data_valid_in(gap_valid), 
        .data_in(gap_out),
        .data_valid_out(d0_valid), 
        .data_out(d0_out)
    );

    // --- Dense 1 (Output) ---
    wire d1_valid;
    wire signed [63:0] d1_out; // 4 classes x 16 bits
    layer_dense1_out #(
        .WEIGHT_FILE("hex/dense1_weights.hex"),
        .BIAS_FILE  ("hex/dense1_bias.hex")
    ) u_dense1 (
        .clk(clk), 
        .reset(reset),
        .data_valid_in(d0_valid), 
        .data_in(d0_out),
        .data_valid_out(d1_valid), 
        .data_out(d1_out)
    );

    // --- Argmax (Simplified combinational find max index) ---
    reg [15:0] predicted_class;
    reg argmax_valid;
    
    always @(posedge clk) begin
        if (reset) begin
            predicted_class <= 0;
            argmax_valid <= 0;
        end else if (d1_valid) begin
            argmax_valid <= 1;
            // Native argmax logic across 4 16-bit signed registers
            if ($signed(d1_out[15:0]) > $signed(d1_out[31:16]) && 
                $signed(d1_out[15:0]) > $signed(d1_out[47:32]) && 
                $signed(d1_out[15:0]) > $signed(d1_out[63:48]))
                predicted_class <= 16'd0;
            else if ($signed(d1_out[31:16]) > $signed(d1_out[47:32]) && 
                     $signed(d1_out[31:16]) > $signed(d1_out[63:48]))
                predicted_class <= 16'd1;
            else if ($signed(d1_out[47:32]) > $signed(d1_out[63:48]))
                predicted_class <= 16'd2;
            else
                predicted_class <= 16'd3;
        end else begin
            argmax_valid <= 0;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            data_valid_out <= 0;
            data_out       <= 0;
        end else begin
            data_valid_out <= argmax_valid;
            data_out       <= predicted_class;
        end
    end
endmodule
