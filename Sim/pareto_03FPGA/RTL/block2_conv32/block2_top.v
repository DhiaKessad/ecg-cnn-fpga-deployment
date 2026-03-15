module block2_top(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [127:0] data_in, // 8 channels * 16 bits = 128 bit output from Block 1
    output reg data_valid_out,
    output reg [511:0] data_out // 32 channels * 16 bits = 512 bit output
);
    // --- Signals ---
    wire [31:0]   conv_valid_bus;
    wire [1023:0] conv_out_bus; // 32 * 32 bit raw data
    
    wire [31:0]   bn_valid_bus;
    wire [511:0]  bn_out_bus; // 32 * 16 bit scaled data
    
    wire [31:0]   pool_valid_bus;
    wire [511:0]  pool_out_bus; // 32 * 16 bit pooled data

    // Since layer_conv_k5 only handles scalar inputs, and we need 8 channels, 
    // a standard conv loop needs to sum the 8 channels inside the module, 
    // OR we need to build a layer_conv_k5x8. 
    // For this simulation structure, we'll assume a dedicated layer_conv_k5x8 exists.
    
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : conv_array
            layer_conv_k5x8 #(
                .WEIGHT_FILE("hex/conv2_weights.hex"), // Note: Weights are [32, 8, 1, 5]
                .BIAS_FILE  ("hex/conv2_bias.hex"),
                .FILTER_ID  (i)
            ) u_conv (
                .clk(clk), 
                .reset(reset),
                .data_valid_in(data_valid_in), 
                .data_in(data_in), // Pass all 8 channels
                .data_valid_out(conv_valid_bus[i]),
                .data_out(conv_out_bus[i*32 +: 32])
            );
            
            layer_batchnorm #(
                .W_BN_FILE("hex/conv2_bn_w.hex"), 
                .B_BN_FILE("hex/conv2_bn_b.hex"),
                .FILTER_ID(i)
            ) u_bn (
                .clk(clk), 
                .reset(reset),
                .data_valid_in(conv_valid_bus[i]), 
                .data_in(conv_out_bus[i*32 +: 32]),
                .data_valid_out(bn_valid_bus[i]), 
                .data_out(bn_out_bus[i*16 +: 16])
            );
            
            layer_maxpool #(
                .POOL_SIZE(2)
            ) u_pool (
                .clk(clk), 
                .reset(reset),
                .data_valid_in(bn_valid_bus[i]), 
                .data_in(bn_out_bus[i*16 +: 16]),
                .data_valid_out(pool_valid_bus[i]), 
                .data_out(pool_out_bus[i*16 +: 16])
            );
        end
    endgenerate

    always @(posedge clk) begin
        if (reset) begin
            data_valid_out <= 0;
            data_out       <= 0;
        end else begin
            data_valid_out <= pool_valid_bus[0];
            data_out       <= pool_out_bus;
        end
    end
endmodule
