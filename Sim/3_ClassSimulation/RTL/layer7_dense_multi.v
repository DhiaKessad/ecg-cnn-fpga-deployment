module layer7_dense_multi #(
    parameter WEIGHT_FILE = "hex/w_weight_dense.hex",
    parameter BIAS_FILE   = "hex/b_bias_dense.hex"
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [1023:0] data_in,
    output reg data_valid_out,
    // 3 classes x 16 bits = 48 bits total
    output reg signed [47:0] data_out
);
    // 64 inputs, 3 outputs. weights[output_idx][input_idx]
    reg signed [15:0] weights [0:191]; // 64 * 3 = 192 elements total
    reg signed [15:0] biases  [0:2];   // 3 biases

    // Combinational MAC
    reg signed [31:0] sum0, sum1, sum2;
    integer i;

    always @(*) begin
        sum0 = {{8{biases[0][15]}}, biases[0], 8'd0};
        sum1 = {{8{biases[1][15]}}, biases[1], 8'd0};
        sum2 = {{8{biases[2][15]}}, biases[2], 8'd0};
        for (i = 0; i < 64; i = i + 1) begin
            // Row-major access: First 64 weights for sum0, etc.
            sum0 = sum0 + ($signed(data_in[i*16 +: 16]) * weights[i*3]);
            sum1 = sum1 + ($signed(data_in[i*16 +: 16]) * weights[i*3 + 1]);
            sum2 = sum2 + ($signed(data_in[i*16 +: 16]) * weights[i*3 + 2]);
        end
    end

    initial begin
        $readmemh(WEIGHT_FILE, weights);
        $readmemh(BIAS_FILE,   biases);
    end

    always @(posedge clk) begin
        if (reset) begin
            data_out       <= 0;
            data_valid_out <= 0;
        end else if (data_valid_in) begin
            data_valid_out  <= 1;
            // Saturation logic for dense outputs
            data_out[15:0]  <= (sum0[31] == 0 && |sum0[30:23]) ? 16'h7FFF : (sum0[31] == 1 && &sum0[30:23] == 0) ? 16'h8000 : sum0[23:8];
            data_out[31:16] <= (sum1[31] == 0 && |sum1[30:23]) ? 16'h7FFF : (sum1[31] == 1 && &sum1[30:23] == 0) ? 16'h8000 : sum1[23:8];
            data_out[47:32] <= (sum2[31] == 0 && |sum2[30:23]) ? 16'h7FFF : (sum2[31] == 1 && &sum2[30:23] == 0) ? 16'h8000 : sum2[23:8];
        end else begin
            data_valid_out <= 0;
        end
    end
endmodule
