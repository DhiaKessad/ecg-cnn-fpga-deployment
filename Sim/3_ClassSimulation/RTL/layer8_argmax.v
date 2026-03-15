module layer8_argmax (
    input clk,
    input reset,
    input data_valid_in,
    // 3 classes x 16 bits
    input signed [47:0] data_in,
    output reg data_valid_out,
    output reg [15:0] data_out // Output padding upper 14 bits with 0s, lowest 2 bits are max class (0, 1, or 2)
);
    wire signed [15:0] val0 = data_in[15:0];
    wire signed [15:0] val1 = data_in[31:16];
    wire signed [15:0] val2 = data_in[47:32];

    always @(posedge clk) begin
        if (reset) begin
            data_valid_out <= 0;
            data_out       <= 0;
        end else if (data_valid_in) begin
            data_valid_out <= 1;
            // Simple hardware Argmax: 
            if (val0 >= val1 && val0 >= val2)
                data_out <= 16'd0;
            else if (val1 >= val0 && val1 >= val2)
                data_out <= 16'd1;
            else
                data_out <= 16'd2;
        end else begin
            data_valid_out <= 0;
        end
    end
endmodule
