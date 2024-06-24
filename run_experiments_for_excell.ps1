# Define the paths for bitlength data and hardware JSON
$bitlength_data_paths = @(
    "./bitlength_data/resnet18_int",
    "./bitlength_data/mobilenetv2_int",
    "./bitlength_data/resnet50_fp",
    "./bitlength_data/vit_fp",
    "./bitlength_data/dlrm_fp",
    "./bitlength_data/gpt2_fp",
    "./bitlength_data/bert_qqp_fp",
    "./bitlength_data/bert_mnli_fp"
)

$hardware_json_paths = @(
    "./hardware_descriptions/snapdragon_888.json",
    "./hardware_descriptions/intel_i7_11700k.json",
    "./hardware_descriptions/nvidia_rtx_3080.json",
    "./hardware_descriptions/intel_xeon_platinum_8380.json",
    "./hardware_descriptions/nvidia_a100.json"
)

# Initialize arrays to store results
$available_datatype_speedups = @()
$arbitrary_bitlength_speedups = @()

# Iterate over each bitlength data path
foreach ($bitlength_data in $bitlength_data_paths) {
    $available_speedup_row = @()
    $arbitrary_speedup_row = @()

    # For each bitlength data path, iterate over each hardware JSON path
    foreach ($hardware_json in $hardware_json_paths) {
        # Construct the command
        $command = "python .\inference_analytical_model.py --software-overhead 0.1 --script-run --batch-size 256 --baseline-bitlength INT8 --data $bitlength_data --hardware-json $hardware_json"
        
        # Run the command and capture the output
        $output = Invoke-Expression $command
        
        # Split the output into the two speedup values
        $speedups = $output -split ","
        $available_speedup = $speedups[0].Trim()
        $arbitrary_speedup = $speedups[1].Trim()

        # Add the speedup values to the respective rows
        $available_speedup_row += $available_speedup
        $arbitrary_speedup_row += $arbitrary_speedup
    }

    # Add the rows to the overall results
    $available_datatype_speedups += $available_speedup_row -join ","
    $arbitrary_bitlength_speedups += $arbitrary_speedup_row -join ","
}

# Output the results for copying into Excel
Write-Output "Available Datatype Speedups:"
$available_datatype_speedups -join "`n"

Write-Output "`nArbitrary Bitlength Speedups:"
$arbitrary_bitlength_speedups -join "`n"