module layer_maxpool#(
    parameter POOL_SIZE = 2 // Stride matches pool size
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  signed [15:0] data_in,
    output reg data_valid_out,
    output reg signed [15:0] data_out
);
    reg signed [15:0] max_val;
    reg [$clog2(POOL_SIZE)-1:0] state;

    always @(posedge clk) begin
        if (reset) begin
            state          <= 0;
            data_valid_out <= 1'b0;
            data_out       <= 16'd0;
            max_val        <= 16'd0;
        end else if (data_valid_in) begin
            if (state == 0) begin
                max_val        <= data_in;
                state          <= state + 1;
                data_valid_out <= 1'b0;
            end else if (state == POOL_SIZE - 1) begin
                // Final element of pool
                data_out       <= (data_in > max_val) ? data_in : max_val;
                state          <= 0;
                data_valid_out <= 1'b1;
            end else begin
                // Intermediate element of pool (if > 2)
                max_val        <= (data_in > max_val) ? data_in : max_val;
                state          <= state + 1;
                data_valid_out <= 1'b0;
            end
        end else begin
            data_valid_out <= 1'b0;
        end
    end
endmodule
