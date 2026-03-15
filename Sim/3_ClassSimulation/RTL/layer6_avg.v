module layer6_avg (
    input  clk,
    input  reset,
    input  data_valid_in,
    input  [1023:0] data_in,
    output reg data_valid_out,
    output reg [1023:0] data_out
);
    reg signed [31:0] acc [0:63];
    reg [5:0] sample_count;
    integer k;

    always @(posedge clk) begin
        if (reset) begin
            sample_count   <= 0;
            data_valid_out <= 0;
            data_out       <= 0;
            for (k = 0; k < 64; k = k + 1) acc[k] <= 0;
        end else if (data_valid_in) begin
            data_valid_out <= 0;

            if (sample_count < 41) begin
                if (sample_count == 40) begin
                    for (k = 0; k < 64; k = k + 1)
                        data_out[k*16 +: 16] <=
                            ($signed(acc[k] + $signed(data_in[k*16 +: 16])) * 800) >>> 15;
                    data_valid_out <= 1;
                    $display("[%0t] L6 Out Valid", $time);
                    sample_count   <= 0;
                    for (k = 0; k < 64; k = k + 1) acc[k] <= 0;
                end else begin
                    // Samples 0–39: just accumulate
                    for (k = 0; k < 64; k = k + 1)
                        acc[k] <= acc[k] + $signed(data_in[k*16 +: 16]);
                    sample_count <= sample_count + 1;
                end
            end
        end else begin
            data_valid_out <= 0;
        end
    end
endmodule