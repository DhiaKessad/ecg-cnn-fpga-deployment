#converting .coe to .hex
#you need .coe for memory wizard in ISE Xilinix
#but you need .hex for simulation
import os

def coe_to_hex(src_folder, dst_folder, bit_width=16):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for filename in os.listdir(src_folder):
        if filename.endswith(".coe"):
            with open(os.path.join(src_folder, filename), 'r') as f:
                lines = f.readlines()

            # Find start of data
            data_start = False
            raw_values = []
            for line in lines:
                if "memory_initialization_vector" in line:
                    data_start = True
                    continue
                if data_start:
                    # Clean line: remove commas, semicolons, and whitespace
                    clean_line = line.replace(',', '').replace(';', '').strip()
                    if clean_line:
                        raw_values.extend(clean_line.split())

            # Convert to Hex (2's complement for negative values)
            hex_values = []
            for val in raw_values:
                int_val = int(val)
                if int_val < 0:
                    int_val = (1 << bit_width) + int_val
                hex_values.append(f"{int_val:04x}") # 4 chars for 16-bit

            # Write to .hex file
            out_filename = filename.replace(".coe", ".hex")
            with open(os.path.join(dst_folder, out_filename), 'w') as f:
                f.write("\n".join(hex_values))

# Usage
coe_to_hex("src_folder", "sim_hex_files")