module layer_dense0 #(
    parameter WEIGHT_FILE = "",
    parameter BIAS_FILE   = ""
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [255:0] data_in, // 16 channels * 16 bits = 256 bits from GAP
    output reg data_valid_out,
    output reg [1023:0] data_out // 64 units * 16 bits = 1024 bits 
);
    // 64 units, 16 inputs.
    // Weights shape: [64, 16] = 1024
    reg signed [15:0] weights [0:1023]; 
    reg signed [15:0] biases  [0:63];

    reg signed [31:0] sum [0:63]; // Accumulators for 64 units
    integer u, i;

    // Combinational MAC for all 64 units across the 16 inputs
    always @(*) begin
        for (u = 0; u < 64; u = u + 1) begin
            sum[u] = {{8{biases[u][15]}}, biases[u], 8'd0}; // Initial bias shifted Q8.8 -> 32
            
            for (i = 0; i < 16; i = i + 1) begin
                // sum[u] = sum + (data_in[i] * weight[u][i])
                sum[u] = sum[u] + ($signed(data_in[i*16 +: 16]) * weights[u*16 + i]);
            end
        end
    end

    initial begin
        if (WEIGHT_FILE != "") $readmemh(WEIGHT_FILE, weights);
        if (BIAS_FILE != "")   $readmemh(BIAS_FILE,   biases);
    end

    always @(posedge clk) begin
        if (reset) begin
            data_out       <= 0;
            data_valid_out <= 0;
        end else if (data_valid_in) begin
            data_valid_out  <= 1;
            // Apply ReLU and Saturation to each of the 64 units
            for (u = 0; u < 64; u = u + 1) begin
                if (sum[u][31]) begin
                    data_out[u*16 +: 16] <= 16'd0; // ReLU
                end else begin
                    // Saturation against Q8.8
                    if (|sum[u][30:23]) 
                        data_out[u*16 +: 16] <= 16'h7FFF;
                    else
                        data_out[u*16 +: 16] <= sum[u][23:8];
                end
            end
        end else begin
            data_valid_out <= 0;
        end
    end
endmodule
