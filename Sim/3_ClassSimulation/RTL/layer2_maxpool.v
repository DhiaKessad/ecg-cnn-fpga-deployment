module layer_maxpool1 (
    input  clk,
    input  reset,
    input  data_valid_in,
    input  signed [15:0] data_in,
    output reg data_valid_out,
    output reg signed [15:0] data_out
);
    reg signed [15:0] first_sample;
    reg state;

    always @(posedge clk) begin
        if (reset) begin
            state          <= 1'b0;
            data_valid_out <= 1'b0;
            data_out       <= 16'd0;
            first_sample   <= 16'd0;
        end else if (data_valid_in) begin
            if (state == 1'b0) begin
                first_sample   <= data_in;
                state          <= 1'b1;
                data_valid_out <= 1'b0;
            end else begin
                data_out       <= (data_in > first_sample) ? data_in : first_sample;
                state          <= 1'b0;
                data_valid_out <= 1'b1;
            end
        end else begin
            data_valid_out <= 1'b0;
        end
    end
endmodule