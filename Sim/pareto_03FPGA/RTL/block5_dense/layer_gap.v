module layer_gap #(
    parameter IN_CHANNELS = 16,
    parameter TIME_STEPS  = 11
)(
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [255:0] data_in, // 16 channels * 16 bits
    output reg data_valid_out,
    output reg [255:0] data_out // 16 channels * 16 bits
);

    reg signed [31:0] sum [0:15];
    reg [5:0] count;
    integer c;
    
    always @(posedge clk) begin
        if (reset) begin
            for (c = 0; c < 16; c = c + 1) sum[c] <= 0;
            count <= 0;
            data_valid_out <= 0;
            data_out <= 0;
        end else if (data_valid_in) begin
            if (count < TIME_STEPS) begin
                for (c = 0; c < 16; c = c + 1) begin
                    sum[c] <= sum[c] + $signed(data_in[c*16 +: 16]);
                end
                count <= count + 1;
                data_valid_out <= 0;
            end 
            
            if (count == TIME_STEPS - 1) begin // Just received the last one
                for (c = 0; c < 16; c = c + 1) begin
                    // Divide by TIME_STEPS (11). 
                    // Approximate integer division: sum * (1/11). 
                    // 1/11 in Q8.8 is ~23 (23/256 = 0.0898)
                    // (sum[c] * 23) >>> 8
                    
                    data_out[c*16 +: 16] <= (sum[c] + $signed(data_in[c*16 +: 16])) / TIME_STEPS;
                end
                data_valid_out <= 1;
                // Wait for reset to process new batch
            end
        end else begin
            data_valid_out <= 0;
        end
    end
endmodule
