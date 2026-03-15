module block3_top(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [511:0] data_in, // 32 channels * 16 bits = 512 bit output from Block 2
    output reg data_valid_out,
    output reg [255:0] data_out // 16 channels * 16 bits = 256 bit output
);
    // --- Signals ---
    wire [15:0]   conv_valid_bus;
    wire [511:0]  conv_out_bus; // 16 * 32 bit raw data
    
    wire [15:0]   bn_valid_bus;
    wire [255:0]  bn_out_bus; // 16 * 16 bit scaled data
    
    wire [15:0]   pool_valid_bus;
    wire [255:0]  pool_out_bus; // 16 * 16 bit pooled data

    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : conv_array
            layer_conv_k5x32 #(
                .WEIGHT_FILE("hex/conv3_weights.hex"), 
                .BIAS_FILE  ("hex/conv3_bias.hex"),
                .FILTER_ID  (i)
            ) u_conv (
                .clk(clk), 
                .reset(reset),
                .data_valid_in(data_valid_in), 
                .data_in(data_in),
                .data_valid_out(conv_valid_bus[i]),
                .data_out(conv_out_bus[i*32 +: 32])
            );
            
            layer_batchnorm #(
                .W_BN_FILE("hex/conv3_bn_w.hex"), 
                .B_BN_FILE("hex/conv3_bn_b.hex"),
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
