module layer_conv2_k3 #(
    parameter WEIGHT_FILE = "hex/w_weight2.hex",
    parameter BIAS_FILE   = "hex/b_bias2.hex",
    parameter FILTER_ID   = 0
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [255:0] data_in,          // 16 channels × 16 bits
    output reg data_valid_out,
    output reg signed [15:0] data_out
);
    reg signed [15:0] window  [0:2][0:15];
    reg signed [15:0] weights [0:1535];  // 32 filters × 3 × 16
    reg signed [15:0] biases  [0:31];
    reg [1:0] count;

    // Combinational MAC
    reg signed [31:0] sum;
    integer m, n;
    always @(*) begin
        sum = {{8{biases[FILTER_ID][15]}}, biases[FILTER_ID], 8'd0}; // Sign-extend and shift left by 8 for Q16.16
        for (m = 0; m < 3; m = m + 1)
            for (n = 0; n < 16; n = n + 1)
                sum = sum + (window[2 - m][n] * weights[FILTER_ID*48 + m*16 + n]);
    end

    integer i, j;

    initial begin
        $readmemh(WEIGHT_FILE, weights);
        $readmemh(BIAS_FILE,   biases);
    end

    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < 3; i = i + 1)
                for (j = 0; j < 16; j = j + 1)
                    window[i][j] <= 0;
            count          <= 0;
            data_valid_out <= 0;
            data_out       <= 0;
        end else if (data_valid_in) begin
            // Shift window
            for (j = 0; j < 16; j = j + 1) begin
                window[2][j] <= window[1][j];
                window[1][j] <= window[0][j];
                window[0][j] <= data_in[j*16 +: 16];
            end

            if (count < 3) count <= count + 1;

            // FIX: window full after 3 inputs → count is 2 before this increment
            data_valid_out <= (count >= 2);

            if (sum[31]) begin
                data_out <= 16'd0;
            end else begin
                if (|sum[30:23])
                    data_out <= 16'h7FFF;
                else
                    data_out <= sum[23:8];
            end
        end else begin
            data_valid_out <= 1'b0;
        end
    end
endmodule