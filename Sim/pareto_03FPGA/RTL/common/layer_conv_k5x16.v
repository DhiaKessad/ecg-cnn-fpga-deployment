module layer_conv_k5x16 #(
    parameter WEIGHT_FILE = "",
    parameter BIAS_FILE   = "",
    parameter FILTER_ID   = 0
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [255:0] data_in, // 16 channels * 16 bits = 256
    output reg data_valid_out,
    output reg signed [31:0] data_out
);
    reg signed [15:0] window  [0:15][0:4];
    // Weights shape: [16 filters, 16 channels, 1, 5] -> 16*16*5 = 1280 elements
    reg signed [15:0] weights [0:1279];
    reg signed [15:0] biases  [0:15];
    reg [3:0] count;
    
    reg signed [31:0] sum;
    integer c, m;
    
    always @(*) begin
        sum = {{16{biases[FILTER_ID][15]}}, biases[FILTER_ID]}; 
        
        for (c = 0; c < 16; c = c + 1) begin
            for (m = 0; m < 5; m = m + 1) begin
                // Filter indexing: [FILTER_ID, channel, 1, time]
                // 3D flattening to 1D: index = FILTER_ID*(16*5) + c*5 + m
                sum = sum + (window[c][4 - m] * weights[FILTER_ID*80 + c*5 + m]);
            end
        end
    end

    initial begin
        if (WEIGHT_FILE != "") $readmemh(WEIGHT_FILE, weights);
        if (BIAS_FILE != "")   $readmemh(BIAS_FILE,   biases);
    end

    always @(posedge clk) begin
        if (reset) begin
            for (c = 0; c < 16; c = c + 1) begin
                window[c][0] <= 0; window[c][1] <= 0; window[c][2] <= 0;
                window[c][3] <= 0; window[c][4] <= 0;
            end
            count          <= 0;
            data_valid_out <= 0;
            data_out       <= 0;
        end else if (data_valid_in) begin
            for (c = 0; c < 16; c = c + 1) begin
                window[c][4] <= window[c][3];
                window[c][3] <= window[c][2];
                window[c][2] <= window[c][1];
                window[c][1] <= window[c][0];
                window[c][0] <= data_in[c*16 +: 16];
            end

            if (count < 5) count <= count + 1;

            if (count >= 4) begin
                data_valid_out <= 1'b1;
                data_out       <= sum;
            end else begin
                data_valid_out <= 1'b0;
            end
        end else begin
            data_valid_out <= 1'b0;
        end
    end
endmodule
