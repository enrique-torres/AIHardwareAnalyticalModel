import json

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