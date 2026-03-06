`timescale 1ns/1ps
module tb_ecg_multi;
    reg  clk;
    reg  reset;
    reg  data_valid_in;
    reg  signed [15:0] data_in;
    wire data_valid_out;
    wire [15:0] data_out;

    reg [15:0] ecg_mem [0:179];
    integer i;

    top_ecg_classifier dut (
        .clk(clk), .reset(reset),
        .data_valid_in(data_valid_in), .data_in(data_in),
        .data_valid_out(data_valid_out), .data_out(data_out)
    );

    initial begin clk = 0; forever #5 clk = ~clk; end

    // Timeout watchdog so simulation doesn't hang
    initial begin
        #500000;
        $display("TIMEOUT — data_valid_out never asserted.");
        $finish;
    end

    initial begin
        $dumpfile("ecg_sim_multi.vcd");
        $dumpvars(0, tb_ecg_multi);

        // Load sample data from file
        $readmemh("hex/input_sample_0.hex", ecg_mem);

        reset = 1;
        data_valid_in = 0;
        data_in = 0;

        #100;
        reset = 0;
        #20;

        $display("--- Starting Multi-Class ECG Simulation ---");
        for (i = 0; i < 180; i = i + 1) begin
            @(posedge clk);
            data_in       <= ecg_mem[i];
            data_valid_in <= 1;
        end
        @(posedge clk);
        data_valid_in <= 0;
        data_in       <= 0;
        
        // Wait for output pulse
        @(posedge data_valid_out);
        @(posedge clk);
        $display("Classification Output (Argmax Class ID): %d", data_out);
        $finish;

    end
endmodule
