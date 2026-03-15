//Processing each filter at a time; kernel 5
module layer_conv1_k5 #(
    parameter WEIGHT_FILE = "hex/w_conv1.hex",
    parameter BIAS_FILE   = "hex/b_conv1.hex",
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

    // Combinational MAC
    reg signed [31:0] sum;
    integer m;
    always @(*) begin
        sum = (biases[FILTER_ID] <<< 8);
        for (m = 0; m < 5; m = m + 1)
            sum = sum + (window[m] * weights[FILTER_ID*5 + m]);
            //full node
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
        end else if (data_valid_in) begin
            // Shift register: window[4] = oldest, window[0] = newest
            //Sliding window method, fast
            window[4] <= window[3];
            window[3] <= window[2];
            window[2] <= window[1];
            window[1] <= window[0];
            window[0] <= data_in;

            if (count < 5) count <= count + 1;

            // Validate if all elements are loaded
            data_valid_out <= (count >= 4);

            // ReLU + scale
            if (sum[31])
                data_out <= 16'd0; //Negative is made 0
            else
                data_out <= sum[23:8];
        end else begin
            data_valid_out <= 1'b0;
        end
    end
endmodule