import json
import csv
import os
from typing import Dict, List

class Layer:
    def __init__(self, name, next_layer, layer_type, num_act_in_gradients, num_act_out_gradients, num_weight_gradients, forward_macs, extra_compute, batch_size):
        self.name = name
        self.next_layer = next_layer
        self.layer_type = layer_type
        self.num_act_in_gradients = num_act_in_gradients
        self.num_act_out_gradients = num_act_out_gradients
        self.num_weight_gradients = num_weight_gradients
        self.forward_macs = forward_macs
        self.extra_compute = extra_compute
        self.batch_size = batch_size
        self._activation_bitlengths = []
        self._weight_bitlengths = []

def read_hardware_json(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def parse_parameters_and_create_layers(args, file_name = "/parameter_values.csv"):
    layers_dict = {}
    with open(args.data + file_name) as csvfile:
        reader = csv.reader(csvfile)
        is_first_row = True
        for row in reader:
            if is_first_row:
                is_first_row = False
                continue
            layer_name = row[0]
            next_layer = row[1]
            layer_type = row[2]
            num_act_in_gradients = int(row[3])
            num_act_out_gradients = int(row[4])
            num_weight_gradients = int(row[5])
            forward_macs = int(row[6])
            extra_compute = int(row[7])

            new_layer = Layer(layer_name, next_layer, layer_type, num_act_in_gradients, num_act_out_gradients, num_weight_gradients, forward_macs, extra_compute, args.batch_size)
            layers_dict[layer_name] = new_layer
    return layers_dict

def parse_bitlengths(folder_path, layers_dict: Dict[str, Layer], acts_file_name = "/activation_bitlengths.csv", wgts_file_name = "/weights_bitlengths.csv"):
    layer_names = []
    with open(folder_path + acts_file_name) as csvfile:
        reader = csv.reader(csvfile)
        is_first_row = True
        for row in reader:
            if is_first_row:
                layer_names = row
                is_first_row = False
                continue
            for i in range(0, len(layer_names)):
                bitlength = float(row[i])
                layers_dict[layer_names[i]]._activation_bitlengths.append(bitlength)

    with open(folder_path + wgts_file_name) as csvfile:
        reader = csv.reader(csvfile)
        is_first_row = True
        for row in reader:
            if is_first_row:
                layer_names = row
                is_first_row = False
                continue
            for i in range(0, len(layer_names)):
                bitlength = float(row[i])
                layers_dict[layer_names[i]]._weight_bitlengths.append(bitlength)

def get_closest_datatype(bitlength, datatypes):
    closest_type = None
    smallest_diff = float('inf')
    for dtype, specs in datatypes.items():
        diff = specs['bitlength'] - bitlength
        if 0 <= diff < smallest_diff:
            smallest_diff = diff
            closest_type = dtype
    return closest_type

def parse_bandwidth(bandwidth_str: str) -> int:
    import re
    
    units = {"B/s": 1, "KB/s": 1024, "MB/s": 1024**2, "GB/s": 1024**3, "TB/s": 1024**4}
    bandwidth_str = bandwidth_str.strip()
    #print(f"Parsing bandwidth string: '{bandwidth_str}'")
    
    for unit in units:
        if bandwidth_str.endswith(unit):
            value_str = re.sub(r'[^\d.]+', '', bandwidth_str.replace(unit, '').strip())
            try:
                value = float(value_str)
                return int(value * units[unit])
            except ValueError:
                raise ValueError(f"Unable to parse the numeric value from {bandwidth_str}")
    
    raise ValueError(f"Unknown bandwidth unit in {bandwidth_str}")

def parse_throughput(throughput_str: str) -> float:
    units = {
        "TOPS": 1e12, "GOPS": 1e9, "MOPS": 1e6, "KOPS": 1e3, "OPS": 1,
        "TFLOPS": 1e12, "GFLOPS": 1e9, "MFLOPS": 1e6, "KFLOPS": 1e3, "FLOPS": 1
    }
    throughput_str = throughput_str.strip().upper()
    #print(f"Parsing throughput string: '{throughput_str}'")  # Debugging

    # Sort units by length in descending order to match the longest unit first
    sorted_units = sorted(units.keys(), key=len, reverse=True)

    for unit in sorted_units:
        if throughput_str.endswith(unit):
            #print(f"Unit string matched: '{unit}'")  # Debugging
            value_str = throughput_str[: -len(unit)].strip()
            #print(f"Extracted value string: '{value_str}'")  # Debugging
            try:
                value = float(value_str)
                #print(f"Converted value: {value}")  # Debugging
                return value * units[unit]
            except ValueError as e:
                #print(f"Error converting '{value_str}' to float: {e}")  # Debugging
                raise ValueError(f"Unable to parse the numeric value from {throughput_str}")
    
    raise ValueError(f"Unknown throughput unit in {throughput_str}")

def calculate_transfer_time(size_in_bits: int, bandwidth: int) -> float:
    if isinstance(bandwidth, str):
        bandwidth = parse_bandwidth(bandwidth)
    return size_in_bits / bandwidth

def simulate_chunked_transfer(size_in_bits: int, source_bandwidth: str, dest_bandwidth: str, chunk_size: int) -> float:
    source_bandwidth_int = parse_bandwidth(source_bandwidth)
    dest_bandwidth_int = parse_bandwidth(dest_bandwidth)
    
    total_transfer_time = 0
    remaining_size = size_in_bits

    while remaining_size > 0:
        chunk = min(remaining_size, chunk_size)
        total_transfer_time += calculate_transfer_time(chunk, source_bandwidth_int)
        total_transfer_time += calculate_transfer_time(chunk, dest_bandwidth_int)
        remaining_size -= chunk

    return total_transfer_time

def simulate_layer(layer: Layer, memory_hierarchy: Dict, datatypes: Dict, baseline_bitlength: int, batch_size: int, compute_units: int, software_overhead: float) -> Dict:
    total_transfer_time_baseline = 0
    total_transfer_time_optimized = 0
    total_transfer_time_arbitrary = 0
    total_compute_time_baseline = 0
    total_compute_time_optimized = 0

    for i, (act_bitlength, wgt_bitlength) in enumerate(zip(layer._activation_bitlengths, layer._weight_bitlengths)):
        # Determine closest available datatype
        act_dtype = get_closest_datatype(act_bitlength, datatypes)
        wgt_dtype = get_closest_datatype(wgt_bitlength, datatypes)

        # Calculate sizes in bits
        act_size_baseline = layer.num_act_in_gradients * baseline_bitlength * batch_size
        wgt_size_baseline = layer.num_weight_gradients * baseline_bitlength
        act_size_optimized = layer.num_act_in_gradients * datatypes[act_dtype]['bitlength'] * batch_size
        wgt_size_optimized = layer.num_weight_gradients * datatypes[wgt_dtype]['bitlength']
        act_size_arbitrary = layer.num_act_in_gradients * act_bitlength * batch_size
        wgt_size_arbitrary = layer.num_weight_gradients * wgt_bitlength

        # Determine which memory level to use based on size and availability
        def get_memory_level(size, hierarchy):
            for level in ['L1_cache', 'L2_cache', 'L3_cache', 'DRAM']:
                if size <= int(hierarchy[level]['size'][:-2]) * 1024:
                    return level
            return 'DRAM'

        act_memory_level_optimized = get_memory_level(act_size_optimized, memory_hierarchy)
        wgt_memory_level_optimized = get_memory_level(wgt_size_optimized, memory_hierarchy)
        act_memory_level_arbitrary = get_memory_level(act_size_arbitrary, memory_hierarchy)
        wgt_memory_level_arbitrary = get_memory_level(wgt_size_arbitrary, memory_hierarchy)

        # Calculate chunked transfer times if needed for optimized datatypes
        if act_memory_level_optimized != 'L1_cache':
            act_transfer_time_optimized = simulate_chunked_transfer(act_size_optimized, memory_hierarchy[act_memory_level_optimized]['bandwidth'], memory_hierarchy['L1_cache']['bandwidth'], int(memory_hierarchy['L1_cache']['size'][:-2]) * 1024)
        else:
            act_transfer_time_optimized = calculate_transfer_time(act_size_optimized, memory_hierarchy['L1_cache']['bandwidth'])

        if wgt_memory_level_optimized != 'L1_cache':
            wgt_transfer_time_optimized = simulate_chunked_transfer(wgt_size_optimized, memory_hierarchy[wgt_memory_level_optimized]['bandwidth'], memory_hierarchy['L1_cache']['bandwidth'], int(memory_hierarchy['L1_cache']['size'][:-2]) * 1024)
        else:
            wgt_transfer_time_optimized = calculate_transfer_time(wgt_size_optimized, memory_hierarchy['L1_cache']['bandwidth'])

        # Calculate chunked transfer times if needed for arbitrary bitlengths
        if act_memory_level_arbitrary != 'L1_cache':
            act_transfer_time_arbitrary = simulate_chunked_transfer(act_size_arbitrary, memory_hierarchy[act_memory_level_arbitrary]['bandwidth'], memory_hierarchy['L1_cache']['bandwidth'], int(memory_hierarchy['L1_cache']['size'][:-2]) * 1024) * (1 + software_overhead)
        else:
            act_transfer_time_arbitrary = calculate_transfer_time(act_size_arbitrary, memory_hierarchy['L1_cache']['bandwidth']) * (1 + software_overhead)

        if wgt_memory_level_arbitrary != 'L1_cache':
            wgt_transfer_time_arbitrary = simulate_chunked_transfer(wgt_size_arbitrary, memory_hierarchy[wgt_memory_level_arbitrary]['bandwidth'], memory_hierarchy['L1_cache']['bandwidth'], int(memory_hierarchy['L1_cache']['size'][:-2]) * 1024) * (1 + software_overhead)
        else:
            wgt_transfer_time_arbitrary = calculate_transfer_time(wgt_size_arbitrary, memory_hierarchy['L1_cache']['bandwidth']) * (1 + software_overhead)

        act_transfer_time_baseline = calculate_transfer_time(act_size_baseline, memory_hierarchy['DRAM']['bandwidth'])
        wgt_transfer_time_baseline = calculate_transfer_time(wgt_size_baseline, memory_hierarchy['DRAM']['bandwidth'])

        # Calculate compute times
        compute_throughput_baseline = parse_throughput(datatypes['FP32']['compute_throughput'])
        compute_throughput_optimized = parse_throughput(datatypes[act_dtype]['compute_throughput'])
        compute_time_baseline = layer.forward_macs / (compute_throughput_baseline * compute_units)
        compute_time_optimized = layer.forward_macs / (compute_throughput_optimized * compute_units)

        total_transfer_time_baseline += act_transfer_time_baseline + wgt_transfer_time_baseline
        total_transfer_time_optimized += act_transfer_time_optimized + wgt_transfer_time_optimized
        total_transfer_time_arbitrary += act_transfer_time_arbitrary + wgt_transfer_time_arbitrary
        total_compute_time_baseline += compute_time_baseline
        total_compute_time_optimized += compute_time_optimized

    return {
        'baseline_transfer_time': total_transfer_time_baseline,
        'optimized_transfer_time': total_transfer_time_optimized,
        'arbitrary_transfer_time': total_transfer_time_arbitrary,
        'baseline_compute_time': total_compute_time_baseline,
        'optimized_compute_time': total_compute_time_optimized
    }

def simulate_network(layers_dict: Dict[str, Layer], memory_hierarchy: Dict, datatypes: Dict, baseline_bitlength: int, batch_size: int, compute_units: int, software_overhead: float) -> Dict:
    total_baseline_transfer_time = 0
    total_optimized_transfer_time = 0
    total_arbitrary_transfer_time = 0
    total_baseline_compute_time = 0
    total_optimized_compute_time = 0

    for layer_name, layer in layers_dict.items():
        print(f"Simulating layer {layer_name}")
        layer_results = simulate_layer(layer, memory_hierarchy, datatypes, baseline_bitlength, batch_size, compute_units, software_overhead)
        total_baseline_transfer_time += layer_results['baseline_transfer_time']
        total_optimized_transfer_time += layer_results['optimized_transfer_time']
        total_arbitrary_transfer_time += layer_results['arbitrary_transfer_time']
        total_baseline_compute_time += layer_results['baseline_compute_time']
        total_optimized_compute_time += layer_results['optimized_compute_time']

    baseline_total_time = total_baseline_transfer_time + total_baseline_compute_time
    optimized_total_time = total_optimized_transfer_time + total_optimized_compute_time
    arbitrary_total_time = total_arbitrary_transfer_time + total_optimized_compute_time

    network_results = {
        'baseline_total_time': baseline_total_time,
        'optimized_total_time': optimized_total_time,
        'arbitrary_total_time': arbitrary_total_time,
        'speedup_optimized': baseline_total_time / optimized_total_time if optimized_total_time > 0 else float('inf'),
        'speedup_arbitrary': baseline_total_time / arbitrary_total_time if arbitrary_total_time > 0 else float('inf')
    }

    return network_results

def list_directories(directory: str) -> List[str]:
    directories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return directories

def select_directory(directory: str) -> str:
    directories = list_directories(directory)
    if not directories:
        raise FileNotFoundError("No directories found in the specified directory.")
    print("Available options:")
    for i, dir_name in enumerate(directories):
        print(f"{i + 1}. {dir_name}")
    choice = int(input("Select an option by number: ")) - 1
    if choice < 0 or choice >= len(directories):
        raise ValueError("Invalid selection.")
    return os.path.join(directory, directories[choice])

def list_hardware_files(directory: str) -> List[str]:
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    return files

def select_hardware_model(directory: str) -> str:
    files = list_hardware_files(directory)
    if not files:
        raise FileNotFoundError("No hardware description files found in the directory.")
    print("Available hardware models:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    choice = int(input("Select a hardware model by number: ")) - 1
    if choice < 0 or choice >= len(files):
        raise ValueError("Invalid selection.")
    return os.path.join(directory, files[choice])

def main(args):    
    if args.hardware_json:
        hardware_model_path = args.hardware_json
    else:
        hardware_model_path = select_hardware_model("hardware_descriptions")

    hardware_model = read_hardware_json(hardware_model_path)
    memory_hierarchy = hardware_model['memory_hierarchy']
    datatypes = hardware_model['datatypes']
    compute_units = hardware_model['compute_units']

    if args.data is None:
        args.data = select_directory("bitlength_data")
    if args.data is None:
        print("The data path argument needs to be provided in order to compute speedups. Exiting now.")
        exit(0)

    layers_dict = parse_parameters_and_create_layers(args)
    parse_bitlengths(args.data, layers_dict)

    baseline_bitlength = args.baseline_bitlength

    results = simulate_network(layers_dict, memory_hierarchy, datatypes, baseline_bitlength, args.batch_size, compute_units, args.software_overhead)

    print(f"Baseline Total Time: {results['baseline_total_time']:.6f}")
    print(f"Optimized Total Time: {results['optimized_total_time']:.6f}")
    print(f"Arbitrary Total Time: {results['arbitrary_total_time']:.6f}")
    print(f"Network Speedup (Optimized): {results['speedup_optimized']:.2f}")
    print(f"Network Speedup (Arbitrary): {results['speedup_arbitrary']:.2f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate neural network performance with different bitlengths and hardware models.")
    parser.add_argument('--data', type=str, default=None, help='Path to the folder containing the network\'s CSV descriptor and the bitlengths.')
    parser.add_argument('--hardware-json', type=str, default=None, help='Path to the JSON file containing the hardware model.')
    parser.add_argument('--baseline-bitlength', type=int, default=32, help='Bitlength used by baseline with which to compare the method against.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size of inputs that will be simulated on the network.')
    parser.add_argument('--software-overhead', type=float, default=0.0, help='Overhead that the method has when compressing/quantizing the values of the network (0.0 - 1.0).')

    args = parser.parse_args()
    main(args)