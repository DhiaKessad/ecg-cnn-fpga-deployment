module layer_conv1_k5 #(
    parameter WEIGHT_FILE = "hex/w_weight1.hex",
    parameter BIAS_FILE   = "hex/b_bias1.hex",
    parameter FILTER_ID   = 0
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  signed [15:0] data_in,
    output reg data_valid_out,
    output reg signed [15:0] data_out
);
    reg signed [15:0] window  [0:4];
    reg signed [15:0] weights [0:79];
    reg signed [15:0] biases  [0:15];
    reg [2:0] count;
    reg [7:0] sample_counter;

    // Combinational MAC
    reg signed [31:0] sum;
    integer m;
    always @(*) begin
        sum = {{8{biases[FILTER_ID][15]}}, biases[FILTER_ID], 8'd0}; // Sign-extend and shift left by 8 for Q16.16
        for (m = 0; m < 5; m = m + 1)
            sum = sum + (window[4 - m] * weights[FILTER_ID*5 + m]);
    end

    initial begin
        $readmemh(WEIGHT_FILE, weights);
        $readmemh(BIAS_FILE,   biases);
    end

    always @(posedge clk) begin
        if (reset) begin
            window[0]     <= 0; window[1] <= 0; window[2] <= 0;
            window[3]     <= 0; window[4] <= 0;
            count         <= 0;
            data_valid_out <= 0;
            data_out      <= 0;
            sample_counter <= 0; // Reset sample_counter
        end else if (data_valid_in) begin
            // Shift register: window[4] = oldest, window[0] = newest
            window[4] <= window[3];
            window[3] <= window[2];
            window[2] <= window[1];
            window[1] <= window[0];
            window[0] <= data_in;

            if (count < 5) count <= count + 1;

            // FIX: window is full after 5 inputs → count reaches 4 before increment
            data_valid_out <= (count >= 4);

            // ReLU + scale
            if (sum[31]) begin
                data_out <= 16'd0;
            end else begin
                // Saturation: if any bit above 23 is 1, max positive
                if (|sum[30:23])
                    data_out <= 16'h7FFF;
                else
                    data_out <= sum[23:8];
            end

            if (data_valid_out && sample_counter < 10) begin
                if (FILTER_ID == 1) begin
                    $display("[%0t] L1 Out Cycle %0d F%0d: %0d | Window: %0d %0d %0d %0d %0d", 
                             $time, sample_counter, FILTER_ID, (sum[31] ? 16'd0 : sum[23:8]),
                             window[0], window[1], window[2], window[3], window[4]);
                end
                sample_counter <= sample_counter + 1;
            end
        end else begin
            data_valid_out <= 1'b0;
        end
    end
endmodule