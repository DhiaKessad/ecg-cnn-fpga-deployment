module layer_dense1_out #(
    parameter WEIGHT_FILE = "",
    parameter BIAS_FILE   = ""
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [1023:0] data_in, // 64 units * 16 bits = 1024 bits 
    output reg data_valid_out,
    output reg signed [63:0] data_out // 4 units * 16 bits = 64
);
    // 4 units, 64 inputs.
    // Weights shape: [4, 64] = 256
    reg signed [15:0] weights [0:255]; 
    reg signed [15:0] biases  [0:3];

    reg signed [31:0] sum [0:3]; // Accumulators for 4 units
    integer u, i;

    // Combinational MAC for all 4 output units
    always @(*) begin
        for (u = 0; u < 4; u = u + 1) begin
            sum[u] = {{8{biases[u][15]}}, biases[u], 8'd0}; 
            
            for (i = 0; i < 64; i = i + 1) begin
                sum[u] = sum[u] + ($signed(data_in[i*16 +: 16]) * weights[u*64 + i]);
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
            // No ReLU on the output linear layer before Softmax/Argmax
            for (u = 0; u < 4; u = u + 1) begin
                // Saturation against Q8.8
                if (sum[u][31] == 0 && |sum[u][30:23]) 
                    data_out[u*16 +: 16] <= 16'h7FFF;
                else if (sum[u][31] == 1 && &sum[u][30:23] == 0)
                    data_out[u*16 +: 16] <= 16'h8000;
                else
                    data_out[u*16 +: 16] <= sum[u][23:8];
            end
        end else begin
            data_valid_out <= 0;
        end
    end
endmodule
