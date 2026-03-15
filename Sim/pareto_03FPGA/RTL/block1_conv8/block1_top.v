module block1_top(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  signed [15:0] data_in, // single channel input
    output reg data_valid_out,
    output reg [127:0] data_out // 8 channels * 16 bits = 128 bit output
);
    // --- Signals ---
    wire [7:0]   conv_valid_bus;
    wire [255:0] conv_out_bus; // 8 * 32 bit raw data (pre-BN)
    
    wire [7:0]   bn_valid_bus;
    wire [127:0] bn_out_bus; // 8 * 16 bit scaled data (post-BN/ReLU)
    
    wire [7:0]   pool_valid_bus;
    wire [127:0] pool_out_bus; // 8 * 16 bit pooled data

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : conv_array
            // 1. Convolution (k=5)
            layer_conv_k5 #(
                .WEIGHT_FILE("hex/conv1_weights.hex"),
                .BIAS_FILE  ("hex/conv1_bias.hex"),
                .FILTER_ID  (i)
            ) u_conv (
                .clk(clk), 
                .reset(reset),
                .data_valid_in(data_valid_in), 
                .data_in(data_in),
                .data_valid_out(conv_valid_bus[i]),
                .data_out(conv_out_bus[i*32 +: 32])
            );
            
            // 2. Batch Normalization + ReLU
            layer_batchnorm #(
                .W_BN_FILE("hex/conv1_bn_w.hex"), 
                .B_BN_FILE("hex/conv1_bn_b.hex"),
                .FILTER_ID(i)
            ) u_bn (
                .clk(clk), 
                .reset(reset),
                .data_valid_in(conv_valid_bus[i]), 
                .data_in(conv_out_bus[i*32 +: 32]),
                .data_valid_out(bn_valid_bus[i]), 
                .data_out(bn_out_bus[i*16 +: 16])
            );
            
            // 3. MaxPooling (pool=2)
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

    // Final output assignment
    // Assuming all parallel filters process synchronously, we take valid from filter 0
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
