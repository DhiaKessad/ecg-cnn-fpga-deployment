module layer_batchnorm #(
    parameter W_BN_FILE = "",
    parameter B_BN_FILE = "",
    parameter FILTER_ID = 0
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  signed [31:0] data_in, // Q16.16 from MAC
    output reg data_valid_out,
    output reg signed [15:0] data_out // Back to 16-bit post-activation
);
    // W_bn and B_bn are the folded BatchNorm parameters:
    // W_bn = gamma / sqrt(variance + epsilon)
    // B_bn = beta - mean * W_bn
    
    // In our offline scripts, we assume these are pre-calculated 
    // and saved as standard Q8.8 inputs to avoid massive FPGA math.
    reg signed [15:0] w_bn [0:63]; // Up to 64 filters
    reg signed [15:0] b_bn [0:63]; 
    
    initial begin
        if (W_BN_FILE != "") $readmemh(W_BN_FILE, w_bn);
        if (B_BN_FILE != "") $readmemh(B_BN_FILE, b_bn);
    end

    // Combinational BN
    reg signed [47:0] scaled;
    reg signed [31:0] shifted;
    
    always @(*) begin
        // data_in is logically 32-bit (Q16.16 equivalent roughly)
        // w_bn is Q8.8
        scaled = data_in * w_bn[FILTER_ID]; 
        
        // shifted = scaled + Bias. Need to align precision.
        // If scaled is large, we slice it back down to a 32-bit realm.
        // Assuming simple alignment:
        shifted = (scaled >>> 8) + {{16{b_bn[FILTER_ID][15]}}, b_bn[FILTER_ID]};
    end

    always @(posedge clk) begin
        if (reset) begin
            data_valid_out <= 0;
            data_out       <= 0;
        end else if (data_valid_in) begin
            data_valid_out <= 1;
            
            // Apply ReLU
            if (shifted[31]) begin // Sign bit is 1 (Negative)
                data_out <= 16'd0;
            end else begin
                // Saturation against 16-bit max (Q8.8 limits)
                // If any upper bit is active, clamp it.
                if (|shifted[30:23]) 
                    data_out <= 16'h7FFF;
                else
                    data_out <= shifted[23:8];
            end
        end else begin
            data_valid_out <= 0;
        end
    end
endmodule
