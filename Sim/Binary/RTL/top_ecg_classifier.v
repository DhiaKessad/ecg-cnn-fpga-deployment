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
                .WEIGHT_FILE("hex/w_conv1.hex"),
                .BIAS_FILE  ("hex/b_conv1.hex"),
                .FILTER_ID  (i)
            ) u_c1 (
                .clk(clk), .reset(reset),
                .data_valid_in(data_valid_in), .data_in(data_in),
                .data_valid_out(c1_valid_bus[i]),
                .data_out(c1_out_bus[i*16 +: 16])
            );
        end
    // A loop of 16 representing 16 filters, the parameter i decides the range of values of w/b
    endgenerate


    // --- Layer 2: MaxPool1 (16 Channels) ---
    //fix later, wide buses
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
                .IN_CH(16),
                .WEIGHT_FILE("hex/w_conv2.hex"),
                .BIAS_FILE  ("hex/b_conv2.hex"),
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
            layer_maxpool1 u_p2 (
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
            layer_conv2_k3 #(
                .IN_CH(32),
                .WEIGHT_FILE("hex/w_conv3.hex"),
                .BIAS_FILE  ("hex/b_conv3.hex"),
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

    // --- Layer 7: Fully Connected ---
    wire fc_valid;
    wire signed [15:0] fc_out;
    layer7_dense #(
        .WEIGHT_FILE("hex/w_dense.hex"),
        .BIAS_FILE  ("hex/b_dense.hex")
    ) u_l7 (
        .clk(clk), .reset(reset),
        .data_valid_in(mean_valid), .data_in(mean_out),
        .data_valid_out(fc_valid), .data_out(fc_out)
    );

    // --- Layer 8: Sigmoid ---
    wire logistic_valid;
    wire [15:0] logistic_out;
    layer8_sigmoid #(
        .LUT_FILE("hex/sigmoid_lut.hex")
    ) u_l8 (
        .clk(clk), .reset(reset),
        .data_valid_in(fc_valid), .data_in(fc_out),
        .data_valid_out(logistic_valid), .data_out(logistic_out)
    );

    // --- Final Output ---
    always @(posedge clk) begin
        if (reset) begin
            data_valid_out <= 1'b0;
            data_out       <= 16'd0;
        end else begin
            data_valid_out <= logistic_valid;
            data_out       <= logistic_out;
        end
    end
endmodule