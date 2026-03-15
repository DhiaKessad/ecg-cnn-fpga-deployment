`timescale 1ns/1ps

module tb_pareto03;

    reg clk;
    reg reset;
    reg data_valid_in;
    reg signed [15:0] data_in;

    wire data_valid_out;
    wire [15:0] data_out; // Predicted class

    // Instantiate Top Module
    top_pareto03 u_dut (
        .clk(clk),
        .reset(reset),
        .data_valid_in(data_valid_in),
        .data_in(data_in),
        .data_valid_out(data_valid_out),
        .data_out(data_out)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Load test data (187 samples)
    reg signed [15:0] ecg_input [0:186];
    integer i;

    initial begin
        clk = 0;
        reset = 1;
        data_valid_in = 0;
        data_in = 0;

        // Initialize with realistic values; we can load the Multi test hex
        // Assuming the file has at least 187 values.
        $readmemh("hex/clean_input.hex", ecg_input);

        $dumpfile("tb_pareto03.vcd");
        $dumpvars(0, tb_pareto03);

        #20 reset = 0;
        #10;

        // Feed data stream sequentially (1 sample per clock)
        for (i = 0; i < 187; i = i + 1) begin
            data_valid_in <= 1;
            data_in <= ecg_input[i];
            #10;
        end
        data_valid_in <= 0;

        // Wait for pipeline to flush and GlobalAveragePooling to complete
        // GAP needs 11 time steps to arrive... Conv limits trim the size down sequentially.
        // Let's give it plenty of time.
        #10000;
        
        $display("======================================================================");
        if (data_valid_out) begin
            $display(">> Paredo_03 DIAGNOSIS COMPLETE <<");
            case (data_out)
                16'd0: $display("Class 0: Normal beat (N)");
                16'd1: $display("Class 1: Supraventricular ectopic beat (S)");
                16'd2: $display("Class 2: Ventricular ectopic beat (V)");
                16'd3: $display("Class 3: Fusion beat (F)");
                default: $display("UNKNOWN CLASS: %d", data_out);
            endcase
            $display("======================================================================");
        end else begin
            $display("[ERROR]: Simulation concluded without a valid output flag.");
        end

        $finish;
    end

    // Monitor output
    always @(posedge clk) begin
        if (data_valid_out) begin
            $display("[%0t] Classification Result: Class %d", $time, data_out);
        end
    end

endmodule
