module top_ecg_classifier(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  signed [15:0] data_in,
    output reg data_valid_out,
    output reg [15:0] data_out
);
    // --- Layer 1: Conv1 (16 Channels) ---
    wire [15:0]  c1_valid_bus;
    wire [255:0] c1_out_bus;
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : conv1_parallel
            layer_conv1_k5 #(
                .WEIGHT_FILE("hex/w_weight1.hex"),
                .BIAS_FILE  ("hex/b_bias1.hex"),
                .FILTER_ID  (i)
            ) u_c1 (
                .clk(clk), .reset(reset),
                .data_valid_in(data_valid_in), .data_in(data_in),
                .data_valid_out(c1_valid_bus[i]),
                .data_out(c1_out_bus[i*16 +: 16])
            );
        end
    endgenerate

    // --- Layer 2: MaxPool1 (16 Channels) ---
    wire [15:0]  p1_valid_bus;
    wire [255:0] p1_out_bus;
    generate
        for (i = 0; i < 16; i = i + 1) begin : pool1_parallel
            layer_maxpool1 u_p1 (
                .clk(clk), .reset(reset),
                .data_valid_in(c1_valid_bus[i]),
                .data_in(c1_out_bus[i*16 +: 16]),
                .data_valid_out(p1_valid_bus[i]),
                .data_out(p1_out_bus[i*16 +: 16])
            );
        end
    endgenerate

    // --- Layer 3: Conv2 (32 Channels) ---
    wire [31:0]  c2_valid_bus;
    wire [511:0] c2_out_bus;
    generate
        for (i = 0; i < 32; i = i + 1) begin : conv2_parallel
            layer_conv2_k3 #(
                .WEIGHT_FILE("hex/w_weight2.hex"),
                .BIAS_FILE  ("hex/b_bias2.hex"),
                .FILTER_ID  (i)
            ) u_c2 (
                .clk(clk), .reset(reset),
                .data_valid_in(p1_valid_bus[0]),
                .data_in(p1_out_bus),
                .data_valid_out(c2_valid_bus[i]),
                .data_out(c2_out_bus[i*16 +: 16])
            );
        end
    endgenerate

    // --- Layer 4: MaxPool2 (32 Channels) ---
    wire [31:0]  p2_valid_bus;
    wire [511:0] p2_out_bus;
    generate
        for (i = 0; i < 32; i = i + 1) begin : pool2_parallel
            layer_maxpool2 u_p2 (
                .clk(clk), .reset(reset),
                .data_valid_in(c2_valid_bus[i]),
                .data_in(c2_out_bus[i*16 +: 16]),
                .data_valid_out(p2_valid_bus[i]),
                .data_out(p2_out_bus[i*16 +: 16])
            );
        end
    endgenerate

    // --- Layer 5: Conv3 (64 Channels) ---
    wire [63:0]   c3_valid_bus;
    wire [1023:0] c3_out_bus;
    generate
        for (i = 0; i < 64; i = i + 1) begin : conv3_parallel
            layer_conv3_k3 #(
                .WEIGHT_FILE("hex/w_weight3.hex"),
                .BIAS_FILE  ("hex/b_bias3.hex"),
                .FILTER_ID  (i)
            ) u_c3 (
                .clk(clk), .reset(reset),
                .data_valid_in(p2_valid_bus[0]),
                .data_in(p2_out_bus),
                .data_valid_out(c3_valid_bus[i]),
                .data_out(c3_out_bus[i*16 +: 16])
            );
        end
    endgenerate

    // --- Layer 6: Global Mean ---
    wire          mean_valid;
    wire [1023:0] mean_out;
    layer6_avg u_l6 (
        .clk(clk), .reset(reset),
        .data_valid_in(c3_valid_bus[0]), .data_in(c3_out_bus),
        .data_valid_out(mean_valid), .data_out(mean_out)
    );

    // --- Layer 7: Fully Connected (Multi-class) ---
    wire fc_valid;
    wire signed [47:0] fc_out;
    layer7_dense_multi #(
        .WEIGHT_FILE("hex/w_weight_dense.hex"),
        .BIAS_FILE  ("hex/b_bias_dense.hex")
    ) u_l7 (
        .clk(clk), .reset(reset),
        .data_valid_in(mean_valid), .data_in(mean_out),
        .data_valid_out(fc_valid), .data_out(fc_out)
    );

    // --- Layer 8: Argmax ---
    wire argmax_valid;
    wire [15:0] argmax_out;
    layer8_argmax u_l8 (
        .clk(clk), .reset(reset),
        .data_valid_in(fc_valid), .data_in(fc_out),
        .data_valid_out(argmax_valid), .data_out(argmax_out)
    );

    // --- Final Output ---
    always @(posedge clk) begin
        if (reset) begin
            data_valid_out <= 1'b0;
            data_out       <= 16'd0;
        end else begin
            data_valid_out <= argmax_valid;
            data_out       <= argmax_out;
        end
    end

    // --- File I/O for Layer Outputs ---
    // synthesis translate_off
    integer fd_c1, fd_p1, fd_c2, fd_p2, fd_c3, fd_mean, fd_fc, fd_argmax;
    initial begin
        fd_c1     = $fopen("layer_output/l1_conv1.txt", "w");
        fd_p1     = $fopen("layer_output/l2_pool1.txt", "w");
        fd_c2     = $fopen("layer_output/l3_conv2.txt", "w");
        fd_p2     = $fopen("layer_output/l4_pool2.txt", "w");
        fd_c3     = $fopen("layer_output/l5_conv3.txt", "w");
        fd_mean   = $fopen("layer_output/l6_mean.txt", "w");
        fd_fc     = $fopen("layer_output/l7_fc.txt", "w");
        fd_argmax = $fopen("layer_output/l8_argmax.txt", "w");
    end

    integer j;
    always @(posedge clk) begin
        if (c1_valid_bus[0]) begin
            for (j = 0; j < 16; j = j + 1)
                $fwrite(fd_c1, "%d ", $signed(c1_out_bus[j*16 +: 16]));
            $fwrite(fd_c1, "\n");
        end
        if (p1_valid_bus[0]) begin
            for (j = 0; j < 16; j = j + 1)
                $fwrite(fd_p1, "%d ", $signed(p1_out_bus[j*16 +: 16]));
            $fwrite(fd_p1, "\n");
        end
        if (c2_valid_bus[0]) begin
            for (j = 0; j < 32; j = j + 1)
                $fwrite(fd_c2, "%d ", $signed(c2_out_bus[j*16 +: 16]));
            $fwrite(fd_c2, "\n");
        end
        if (p2_valid_bus[0]) begin
            for (j = 0; j < 32; j = j + 1)
                $fwrite(fd_p2, "%d ", $signed(p2_out_bus[j*16 +: 16]));
            $fwrite(fd_p2, "\n");
        end
        if (c3_valid_bus[0]) begin
            for (j = 0; j < 64; j = j + 1)
                $fwrite(fd_c3, "%d ", $signed(c3_out_bus[j*16 +: 16]));
            $fwrite(fd_c3, "\n");
        end
        if (mean_valid) begin
            for (j = 0; j < 64; j = j + 1)
                $fwrite(fd_mean, "%d ", $signed(mean_out[j*16 +: 16]));
            $fwrite(fd_mean, "\n");
        end
        if (fc_valid) begin
            for (j = 0; j < 3; j = j + 1)
                $fwrite(fd_fc, "%d ", $signed(fc_out[j*16 +: 16]));
            $fwrite(fd_fc, "\n");
        end
        if (argmax_valid) begin
            $fwrite(fd_argmax, "%d\n", $signed(argmax_out));
        end
    end
    // synthesis translate_on

endmodule