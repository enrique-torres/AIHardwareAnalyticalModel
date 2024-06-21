import json
from .menu_helpers import *

# Determine which memory level to use based on size and availability
def get_memory_level(size, hierarchy, layer_depth):
    # If we are on the first layer, we are always getting our inputs from DRAM
    if layer_depth == 0:
        return 'DRAM', hierarchy['DRAM']['size'] - size
    for level in ['L1_cache', 'L2_cache', 'L3_cache', 'DRAM']:
        if level in hierarchy.keys() and size < hierarchy[level]['size']:
            return level, hierarchy[level]['size'] - size
    print("ERROR: Not enough DRAM memory for model. Exiting.")
    exit(1)

def read_hardware_json(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def parse_bandwidth(bandwidth_str: str) -> int:
    import re
    
    units = {"KB/s": 1024, "MB/s": 1024**2, "GB/s": 1024**3, "TB/s": 1024**4}
    bandwidth_str = bandwidth_str.strip()
    
    for unit in units:
        if bandwidth_str.endswith(unit):
            value_str = re.sub(r'[^\d.]+', '', bandwidth_str.replace(unit, '').strip())
            try:
                value = float(value_str)
                return int(value * units[unit])
            except ValueError:
                raise ValueError(f"Unable to parse the numeric value from {bandwidth_str}")
    
    raise ValueError(f"Unknown bandwidth unit in {bandwidth_str}")

def parse_size(size_str: str) -> int:
    import re
    
    units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    size_str = size_str.strip()
    
    for unit in units:
        if size_str.endswith(unit):
            value_str = re.sub(r'[^\d.]+', '', size_str.replace(unit, '').strip())
            try:
                value = float(value_str)
                return int(value * units[unit])
            except ValueError:
                raise ValueError(f"Unable to parse the numeric value from {size_str}")
    
    raise ValueError(f"Unknown size unit in {size_str}")

def parse_throughput(throughput_str: str) -> float:
    units = {
        "TOPS": 1e12, "GOPS": 1e9, "MOPS": 1e6, "KOPS": 1e3, "OPS": 1,
        "TFLOPS": 1e12, "GFLOPS": 1e9, "MFLOPS": 1e6, "KFLOPS": 1e3, "FLOPS": 1
    }
    throughput_str = throughput_str.strip().upper()

    # Sort units by length in descending order to match the longest unit first
    sorted_units = sorted(units.keys(), key=len, reverse=True)

    for unit in sorted_units:
        if throughput_str.endswith(unit):
            value_str = throughput_str[: -len(unit)].strip()
            try:
                value = float(value_str)
                return value * units[unit]
            except ValueError as e:
                raise ValueError(f"Unable to parse the numeric value from {throughput_str}")
    
    raise ValueError(f"Unknown throughput unit in {throughput_str}")

def parse_datatype(datatype: str):
    units = {
        "FP64":(64, "float"), "FP32": (32, "float"), "FP16": (16, "float"), "BF16": (16, "float"), "FP8": (8, "float"), "FP4": (4, "float"),
        "INT64": (64, "integer"), "INT32": (32, "integer"), "INT16": (16, "integer"), "INT8": (8, "integer"), "INT4": (4, "integer")
    }
    if datatype in units.keys():
        return units[datatype]
    else:
        print(f"Unrecognized datatype selected for conversion. Error. Exiting now.")
        exit(1)

def load_hardware_model(hardware_json_path):
    if hardware_json_path:
        hardware_model_path = hardware_json_path
    else:
        hardware_model_path = select_hardware_model("hardware_descriptions")

    hardware_model = read_hardware_json(hardware_model_path)
    memory_hierarchy = hardware_model['memory_hierarchy']

    # Convert size and bandwith strings to standardized integer values
    for memory_level in memory_hierarchy.keys():
        memory_hierarchy[memory_level]['size'] = parse_size(memory_hierarchy[memory_level]['size'])
        memory_hierarchy[memory_level]['bandwith'] = parse_bandwidth(memory_hierarchy[memory_level]['bandwidth'])
    
    datatypes = hardware_model['datatypes']

    # Convert throughput strings to standardized integer values
    for datatype in datatypes.keys():
        datatypes[datatype]['compute_throughput'] = parse_throughput(datatypes[datatype]['compute_throughput'])

    compute_units = hardware_model['compute_units']

    return memory_hierarchy, datatypes, compute_units