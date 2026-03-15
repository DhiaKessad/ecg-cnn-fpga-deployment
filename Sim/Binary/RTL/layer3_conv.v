module layer_conv2_k3 #(
    parameter IN_CH = 16,
    parameter WEIGHT_FILE = "hex/w_conv2.hex",
    parameter BIAS_FILE   = "hex/b_conv2.hex",
    parameter FILTER_ID   = 0
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [IN_CH*16-1:0] data_in,          // 16 channels × 16 bits
    output reg data_valid_out,
    output reg signed [15:0] data_out
);
    reg signed [15:0] window  [0:2][0:IN_CH-1];
    reg signed [15:0] weights [0:1535];  // 32 filters × 3 × 16
    reg signed [15:0] biases  [0:31];
    reg [1:0] count;

    // Combinational MAC
    reg signed [31:0] sum;
    integer i, j;
    always @(*) begin
        sum = (biases[FILTER_ID] <<< 8);
        for (i = 0; i < 3; i = i + 1)
            for (j = 0; j < IN_CH; j = j + 1)
                sum = sum + (window[i][j] * weights[FILTER_ID*(3*IN_CH) + i*IN_CH + j]);
    end

    initial begin
        $readmemh(WEIGHT_FILE, weights);
        $readmemh(BIAS_FILE,   biases);
    end

    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < 3; i = i + 1)
                for (j = 0; j < IN_CH; j = j + 1)
                    window[i][j] <= 0;
            count          <= 0;
            data_valid_out <= 0;
            data_out       <= 0;
        end else if (data_valid_in) begin
            // Shift window
            for (j = 0; j < IN_CH; j = j + 1) begin
                window[2][j] <= window[1][j];
                window[1][j] <= window[0][j];
                window[0][j] <= data_in[j*16 +: 16];
            end

            if (count < 3) count <= count + 1;

            // FIX: window full after 3 inputs → count is 2 before this increment
            data_valid_out <= (count >= 2);

            if (sum[31])
                data_out <= 16'd0;
            else
                data_out <= sum[23:8];
        end else begin
            data_valid_out <= 1'b0;
        end
    end
endmodule