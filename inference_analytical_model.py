import math
from typing import Dict

from core.hardware_description_helpers import *
from core.data_helpers import *

def calculate_transfer_time(size_in_bits: int, source_bandwith: int, destination_bandwith: int) -> float:
    if isinstance(source_bandwith, str):
        source_bandwith = parse_bandwidth(source_bandwith)
    if isinstance(destination_bandwith, str):
        destination_bandwith = parse_bandwidth(destination_bandwith)
    # Always use the more restrictive bandwith as the bottleneck for memory transfers
    if source_bandwith > destination_bandwith:
        bandwidth = destination_bandwith
    else:
        bandwidth = source_bandwith
    return size_in_bits / bandwidth

def calculate_time_weights_L1(layer: Layer, memory_hierarchy: Dict, datatypes: Dict, compute_units: int, compute_bitlength: str, software_overhead: float, act_in_size: int, act_out_size: int, wgt_size: int, act_in_memory_level: str, wgt_memory_level: str):
    # First simulate the writing of output activations from previous layer, except in the first layer's case. We also assume that the last layer's output activations are written to DRAM
    if layer.previous_layer != None:
        prev_out_act_write_time = calculate_transfer_time(act_in_size, memory_hierarchy['L1_cache']['bandwidth'], memory_hierarchy[act_in_memory_level]['bandwidth'])
    else:
        prev_out_act_write_time = 0
    if layer.next_layer == None:
        out_act_write_time = calculate_transfer_time(act_out_size, memory_hierarchy['L1_cache']['bandwidth'], memory_hierarchy['DRAM']['bandwidth'])
    else:
        out_act_write_time = 0
    # Then we simulate reading the weights of the layer from DRAM to L1
    wgt_read_time = calculate_transfer_time(wgt_size, memory_hierarchy['DRAM']['bandwidth'], memory_hierarchy[wgt_memory_level]['bandwidth'])
    # Then we simulate reading the input activations from the level they are at to L1
    act_in_read_time = calculate_transfer_time(act_in_size, memory_hierarchy[act_in_memory_level]['bandwidth'], memory_hierarchy['L1_cache']['bandwidth'])
    # Finally we also need to calculate the compute time for the layer. We can assume that that while we have to wait for the output activations from the previous layer to be written,
    # memory transfers and compute can happen in parallel. For this reason, our total time will be the maximum between all the read transfers and the compute time, plus the write time
    compute_time = layer.forward_macs / (datatypes[compute_bitlength]['compute_throughput'] * compute_units) * (1 + software_overhead)
    transfer_time = prev_out_act_write_time + out_act_write_time + wgt_read_time + act_in_read_time
    total_time = prev_out_act_write_time + out_act_write_time + max(wgt_read_time + act_in_read_time, compute_time)
    return total_time, transfer_time, compute_time

def calculate_time_weights_not_L1(layer: Layer, memory_hierarchy: Dict, datatypes: Dict, compute_units: int, compute_bitlength: str, software_overhead: float, act_in_size: int, act_out_size: int, wgt_size: int, act_in_memory_level: str, wgt_memory_level: str):
    # First simulate the writing of output activations from previous layer, except in the first layer's case. We also assume that the last layer's output activations are written to DRAM
    if layer.previous_layer != None:
        prev_out_act_write_time = calculate_transfer_time(act_in_size, memory_hierarchy['L1_cache']['bandwidth'], memory_hierarchy[act_in_memory_level]['bandwidth'])
    else:
        prev_out_act_write_time = 0
    if layer.next_layer == None:
        out_act_write_time = calculate_transfer_time(act_out_size, memory_hierarchy['L1_cache']['bandwidth'], memory_hierarchy['DRAM']['bandwidth'])
    else:
        out_act_write_time = 0
    
    # Then we have to decide whether to mini-batch weights or activations. If we minibatch activations, we need to read all the weights for each chunk of activations.
    # If we decide to mini-batch weights, we need to read all activations for each chunk of weights. As such, we optimize transfers by mini-batching the larger
    # footprint generator of the two
    if act_in_size > wgt_size:
        # Then we simulate reading the input activations from the level they are at to L1
        act_in_read_time = calculate_transfer_time(act_in_size, memory_hierarchy[act_in_memory_level]['bandwidth'], memory_hierarchy['L1_cache']['bandwidth'])
        total_activation_size = act_in_size + act_out_size
        num_chunks_acts = math.ceil(total_activation_size / memory_hierarchy['L1_cache']['size'])
        #print(f"For layer {layer.name}, weight size: {wgt_size}, we require this many activation chunks: {num_chunks_acts} from level {act_in_memory_level}")
        # And then we calculate the weights reading time, which will be the equivalent for the size of the weights times the number of chunks of activations
        wgt_read_time = calculate_transfer_time(wgt_size * num_chunks_acts, memory_hierarchy[wgt_memory_level]['bandwidth'], memory_hierarchy['L1_cache']['bandwidth'])
    else:
        wgt_read_time = calculate_transfer_time(wgt_size, memory_hierarchy[wgt_memory_level]['bandwidth'], memory_hierarchy['L1_cache']['bandwidth'])
        num_chunks_wgts = math.ceil(wgt_size / memory_hierarchy['L1_cache']['size'])
        #print(f"For layer {layer.name}, activation size: {act_in_size}, we require this many weight chunks: {num_chunks_wgts} from level {wgt_memory_level}")
        act_in_read_time = calculate_transfer_time(act_in_size * num_chunks_wgts, memory_hierarchy[act_in_memory_level]['bandwidth'], memory_hierarchy['L1_cache']['bandwidth'])
    
    # Finally we also need to calculate the compute time for the layer. We can assume that that while we have to wait for the output activations from the previous layer to be written,
    # memory transfers and compute can happen in parallel. For this reason, our total time will be the maximum between all the read transfers and the compute time, plus the write time
    compute_time = layer.forward_macs / (datatypes[compute_bitlength]['compute_throughput'] * compute_units) * (1 + software_overhead)
    transfer_time = prev_out_act_write_time + out_act_write_time + wgt_read_time + act_in_read_time
    total_time = prev_out_act_write_time + out_act_write_time + max(wgt_read_time + act_in_read_time, compute_time)
    return total_time, transfer_time, compute_time

def simulate_layer(layer: Layer, layer_depth: int, memory_hierarchy: Dict, datatypes: Dict, baseline_bitlength: str, batch_size: int, compute_units: int, software_overhead: float) -> Dict:
    total_transfer_time_baseline = 0
    total_transfer_time_optimized = 0
    total_transfer_time_arbitrary = 0
    total_compute_time_baseline = 0
    total_compute_time_optimized = 0
    total_compute_time_arbitrary = 0
    total_time_baseline = 0
    total_time_optimized = 0
    total_time_arbitrary = 0

    baseline_bitlength_int, baseline_datatype = parse_datatype(baseline_bitlength)

    for bitlength_index, (act_bitlength, wgt_bitlength) in enumerate(zip(layer._activation_bitlengths, layer._weight_bitlengths)):
        # Determine closest available datatype for output activations and for weights
        act_out_dtype = get_closest_datatype(act_bitlength, datatypes, layer.network_metadata)
        wgt_dtype = get_closest_datatype(wgt_bitlength, datatypes, layer.network_metadata)
        compute_dtype = get_closest_datatype(max(parse_datatype(act_out_dtype)[0], parse_datatype(wgt_dtype)[0]), datatypes, layer.network_metadata)

        # Determine closest available datatype for input activations
        if layer.previous_layer != None:
            act_in_dtype = get_closest_datatype(layer.previous_layer._activation_bitlengths[bitlength_index], datatypes, layer.network_metadata)
        else:
            act_in_dtype = act_out_dtype
        
        # Calculate sizes of input and output activations in bits
        act_in_size_baseline = layer.num_act_in_gradients * baseline_bitlength_int * batch_size
        act_in_size_optimized = layer.num_act_in_gradients * datatypes[act_in_dtype]['bitlength'] * batch_size
        if layer.previous_layer != None:
            act_in_size_arbitrary = layer.num_act_in_gradients * layer.previous_layer._activation_bitlengths[bitlength_index] * batch_size
        else:
            act_in_size_arbitrary = layer.num_act_in_gradients * act_bitlength * batch_size
        act_out_size_baseline = layer.num_act_out_gradients * baseline_bitlength_int * batch_size
        act_out_size_optimized = layer.num_act_out_gradients * datatypes[act_out_dtype]['bitlength'] * batch_size
        act_out_size_arbitrary = layer.num_act_out_gradients * act_bitlength * batch_size

        # Calculate total sizes in bits
        wgt_size_baseline = layer.num_weight_gradients * baseline_bitlength_int
        wgt_size_optimized = layer.num_weight_gradients * datatypes[wgt_dtype]['bitlength']
        wgt_size_arbitrary = layer.num_weight_gradients * wgt_bitlength

        # First we read the weights for the layer, which we'll keep on the closest memory level to the compute units possible
        wgt_memory_level_baseline, remaining_level_space_after_wgt_baseline = get_memory_level(wgt_size_baseline, memory_hierarchy, layer_depth)
        wgt_memory_level_optimized, remaining_level_space_after_wgt_optimized = get_memory_level(wgt_size_optimized, memory_hierarchy, layer_depth)
        wgt_memory_level_arbitrary, remaining_level_space_after_wgt_arbitrary = get_memory_level(wgt_size_arbitrary, memory_hierarchy, layer_depth)

        # Then we need to mind the space left on that hierarchy level when reading input activations and writing output activations
        memory_hierarchy_updated_baseline = dict(memory_hierarchy)
        memory_hierarchy_updated_baseline[wgt_memory_level_baseline]['size'] = remaining_level_space_after_wgt_baseline
        memory_hierarchy_updated_optimized = dict(memory_hierarchy)
        memory_hierarchy_updated_optimized[wgt_memory_level_optimized]['size'] = remaining_level_space_after_wgt_optimized
        memory_hierarchy_updated_arbitrary = dict(memory_hierarchy)
        memory_hierarchy_updated_arbitrary[wgt_memory_level_arbitrary]['size'] = remaining_level_space_after_wgt_arbitrary

        # Now we need to do the same thing but for activations, keeping in mind that we need to handle both input activations on-chip, as well as output activations
        # If it's the first layer's inputs, we always ready them from DRAM
        if layer_depth != 0:
            act_in_memory_level_baseline, _ = get_memory_level(act_in_size_baseline, memory_hierarchy_updated_baseline, layer_depth)
            act_in_memory_level_optimized, _ = get_memory_level(act_in_size_optimized, memory_hierarchy_updated_optimized, layer_depth)
            act_in_memory_level_arbitrary, _ = get_memory_level(act_in_size_arbitrary, memory_hierarchy_updated_arbitrary, layer_depth)
        else:
            act_in_memory_level_baseline = 'DRAM'
            act_in_memory_level_optimized = 'DRAM'
            act_in_memory_level_arbitrary = 'DRAM'

        # Now we have to check whether the weights fit in L1. There are two main scenarios: weights for the layer fit in the lowest memory level or they don't
        # In the first scenario, we read input activations from where we left the output activations of the previous layer. We then generate output activations for
        # this layer and we store them in the lowest level possible for the next layer to read them from. We assume we can schedule this beforehand for best performance. 
        # In the second scenario weights don't fit in L1. In this case, input activations have to be read in chunks such that both input and output activations from that
        # chunk fit in L1. Input activations will be read once again from the level the previous layer left them, and output activations will be written to the best possible
        # level for the next layer to read from. Finally, per chunk of input activations that is read to L1, all weights from the layer will be read to compute output
        # activations for that chunk of input activations. For simplicity purposes, we simulate the writing of output activations on the next layer, since then we can
        # simulate as if the hardware was scheduled for optimal data transfers.
        #L1_size = memory_hierarchy['L1_cache']['size']
        if wgt_memory_level_baseline == 'L1_cache':
            total_time_baseline, transfer_time_baseline, compute_time_baseline = calculate_time_weights_L1(layer, memory_hierarchy_updated_baseline, datatypes, 
                                                                                                           compute_units, baseline_bitlength, software_overhead, 
                                                                                                           act_in_size_baseline, act_out_size_baseline, wgt_size_baseline, 
                                                                                                           act_in_memory_level_baseline, wgt_memory_level_baseline)
            #print(f"Weights fit on L1 for baseline of layer {layer.name}. L1 size: {L1_size}. Weights size: {wgt_size_baseline}. Remaining space: {L1_size - wgt_size_baseline}")
        else:
            total_time_baseline, transfer_time_baseline, compute_time_baseline = calculate_time_weights_not_L1(layer, memory_hierarchy_updated_baseline, datatypes, 
                                                                                                           compute_units, baseline_bitlength, software_overhead, 
                                                                                                           act_in_size_baseline, act_out_size_baseline, wgt_size_baseline, 
                                                                                                           act_in_memory_level_baseline, wgt_memory_level_baseline)
            #print(f"Weights don't fit on L1 for baseline of layer {layer.name}. L1 size: {L1_size}. Weights size: {wgt_size_baseline}. Selected weights level: {wgt_memory_level_baseline}")
        if wgt_memory_level_optimized == 'L1_cache':
            total_time_optimized, transfer_time_optimized, compute_time_optimized = calculate_time_weights_L1(layer, memory_hierarchy_updated_optimized, datatypes, 
                                                                                                              compute_units, compute_dtype, software_overhead, 
                                                                                                              act_in_size_optimized, act_out_size_optimized, wgt_size_optimized, 
                                                                                                              act_in_memory_level_optimized, wgt_memory_level_optimized)
            #print(f"Weights fit on L1 for optimized of layer {layer.name}. L1 size: {L1_size}. Weights size: {wgt_size_optimized}. Remaining space: {L1_size - wgt_size_optimized}")
        else:
            total_time_optimized, transfer_time_optimized, compute_time_optimized = calculate_time_weights_not_L1(layer, memory_hierarchy_updated_optimized, datatypes, 
                                                                                                              compute_units, compute_dtype, software_overhead, 
                                                                                                              act_in_size_optimized, act_out_size_optimized, wgt_size_optimized, 
                                                                                                              act_in_memory_level_optimized, wgt_memory_level_optimized)
            #print(f"Weights don't fit on L1 for optimized of layer {layer.name}. L1 size: {L1_size}. Weights size: {wgt_size_optimized}. Selected weights level: {wgt_memory_level_optimized}")
        if wgt_memory_level_arbitrary == 'L1_cache':
            # On chip, the arbitrary bitlengths will be converted to available datatypes by software/hardware compressor/decompressor units. As such, unless our architecture is bit-serial
            # we have to use the "available" datatypes instead of arbitrary for on-chip transfers
            if act_in_memory_level_arbitrary != 'DRAM':
                act_in_real_transfer_data = act_in_size_optimized
                act_in_real_memory_level = act_in_memory_level_optimized
            else:
                act_in_real_transfer_data = act_in_size_arbitrary
                act_in_real_memory_level = act_in_memory_level_arbitrary
            if wgt_memory_level_arbitrary != 'DRAM':
                wgt_real_transfer_data = wgt_size_optimized
                wgt_real_memory_level = wgt_memory_level_optimized
            else:
                wgt_real_transfer_data = wgt_size_arbitrary
                wgt_real_memory_level = wgt_memory_level_arbitrary
            
            total_time_arbitrary, transfer_time_arbitrary, compute_time_arbitrary = calculate_time_weights_L1(layer, memory_hierarchy_updated_arbitrary, datatypes, 
                                                                                                              compute_units, compute_dtype, software_overhead, 
                                                                                                              act_in_real_transfer_data, act_out_size_arbitrary, wgt_real_transfer_data, 
                                                                                                              act_in_real_memory_level, wgt_real_memory_level)
            #print(f"Weights fit on L1 for arbitrary of layer {layer.name}. L1 size: {L1_size}. Weights size: {wgt_size_arbitrary}. Remaining space: {L1_size - wgt_size_arbitrary}")
        else:
            # On chip, the arbitrary bitlengths will be converted to available datatypes by software/hardware compressor/decompressor units. As such, unless our architecture is bit-serial
            # we have to use the "available" datatypes instead of arbitrary for on-chip transfers
            if act_in_memory_level_arbitrary != 'DRAM':
                act_in_real_transfer_data = act_in_size_optimized
                act_in_real_memory_level = act_in_memory_level_optimized
            else:
                act_in_real_transfer_data = act_in_size_arbitrary
                act_in_real_memory_level = act_in_memory_level_arbitrary
            if wgt_memory_level_arbitrary != 'DRAM':
                wgt_real_transfer_data = wgt_size_optimized
                wgt_real_memory_level = wgt_memory_level_optimized
            else:
                wgt_real_transfer_data = wgt_size_arbitrary
                wgt_real_memory_level = wgt_memory_level_arbitrary

            total_time_arbitrary, transfer_time_arbitrary, compute_time_arbitrary = calculate_time_weights_not_L1(layer, memory_hierarchy_updated_arbitrary, datatypes, 
                                                                                                              compute_units, compute_dtype, software_overhead, 
                                                                                                              act_in_real_transfer_data, act_out_size_arbitrary, wgt_real_transfer_data, 
                                                                                                              act_in_real_memory_level, wgt_real_memory_level)
            #print(f"Weights don't fit on L1 for arbitrary of layer {layer.name}. L1 size: {L1_size}. Weights size: {wgt_size_arbitrary}. Selected weights level: {wgt_memory_level_arbitrary}")

        total_transfer_time_baseline += transfer_time_baseline
        total_transfer_time_optimized += transfer_time_optimized
        total_transfer_time_arbitrary += transfer_time_arbitrary
        total_compute_time_baseline += compute_time_baseline
        total_compute_time_optimized += compute_time_optimized
        total_compute_time_arbitrary += compute_time_arbitrary
        total_time_baseline += total_time_baseline
        total_time_optimized += total_time_optimized
        total_time_arbitrary += total_time_arbitrary

    return {
        'baseline_transfer_time': total_transfer_time_baseline,
        'optimized_transfer_time': total_transfer_time_optimized,
        'arbitrary_transfer_time': total_transfer_time_arbitrary,
        'baseline_compute_time': total_compute_time_baseline,
        'optimized_compute_time': total_compute_time_optimized,
        'arbitrary_compute_time': total_compute_time_arbitrary,
        'baseline_total_time': total_time_baseline,
        'optimized_total_time': total_time_optimized,
        'arbitrary_total_time': total_time_arbitrary
    }

def simulate_network(layers_dict: Dict[str, Layer], memory_hierarchy: Dict, datatypes: Dict, baseline_bitlength: str, batch_size: int, compute_units: int, software_overhead: float) -> Dict:
    total_baseline_transfer_time = 0
    total_optimized_transfer_time = 0
    total_arbitrary_transfer_time = 0
    total_baseline_compute_time = 0
    total_optimized_compute_time = 0
    total_arbitrary_compute_time = 0
    total_baseline_time = 0
    total_optimized_time = 0
    total_arbitrary_time = 0

    for layer_depth, (layer_name, layer) in enumerate(layers_dict.items()):
        #print(f"Simulating layer {layer_name}")
        layer_results = simulate_layer(layer, layer_depth, memory_hierarchy, datatypes, baseline_bitlength, batch_size, compute_units, software_overhead)
        total_baseline_transfer_time += layer_results['baseline_transfer_time']
        total_optimized_transfer_time += layer_results['optimized_transfer_time']
        total_arbitrary_transfer_time += layer_results['arbitrary_transfer_time']
        total_baseline_compute_time += layer_results['baseline_compute_time']
        total_optimized_compute_time += layer_results['optimized_compute_time']
        total_arbitrary_compute_time += layer_results['arbitrary_compute_time']
        total_baseline_time += layer_results['baseline_total_time']
        total_optimized_time += layer_results['optimized_total_time']
        total_arbitrary_time += layer_results['arbitrary_total_time']

    network_results = {
        'baseline_total_time': total_baseline_time,
        'optimized_total_time': total_optimized_time,
        'arbitrary_total_time': total_arbitrary_time,
        'speedup_optimized': total_baseline_time / total_optimized_time if total_optimized_time > 0 else float('inf'),
        'speedup_arbitrary': total_baseline_time / total_arbitrary_time if total_arbitrary_time > 0 else float('inf'),
        'compute_speedup_optimized': total_baseline_compute_time / total_optimized_compute_time if total_optimized_compute_time > 0 else float('inf'),
        'compute_speedup_arbitrary': total_baseline_compute_time / total_arbitrary_compute_time if total_arbitrary_compute_time > 0 else float('inf'),
        'memory_speedup_optimized': total_baseline_transfer_time / total_optimized_transfer_time if total_optimized_transfer_time > 0 else float('inf'),
        'memory_speedup_arbitrary': total_baseline_transfer_time / total_arbitrary_transfer_time if total_arbitrary_transfer_time > 0 else float('inf')
    }

    return network_results

def main(args):    
    # Load the network information and the bitlengths produced by the compression/optimization method
    layers_dict = load_network_and_bitlengths(args.data, args.batch_size)

    # Load the hardware model with its memory hierarchy, its parsed available datatypes and the number of compute units
    memory_hierarchy, datatypes, compute_units = load_hardware_model(args.hardware_json)

    # Simulate a forward pass on the network, with memory transfers and compute unit throughput to calculate estimated timings for baseline, available datatypes and arbitrary datatypes
    results = simulate_network(layers_dict, memory_hierarchy, datatypes, args.baseline_bitlength, args.batch_size, compute_units, args.software_overhead)

    # Print the results to terminal
    if args.script_run:
        print(f"{results['speedup_optimized']:.2f},{results['speedup_arbitrary']:.2f}")
    else:
        print(f"Baseline Total Time: {results['baseline_total_time']:.6f}")
        print(f"Optimized Total Time: {results['optimized_total_time']:.6f}")
        print(f"Arbitrary Total Time: {results['arbitrary_total_time']:.6f}")
        print(f"Network Purely Compute Speedup (Optimized): {results['compute_speedup_optimized']:.2f}")
        print(f"Network Purely Compute Speedup (Arbitrary): {results['compute_speedup_arbitrary']:.2f}")
        print(f"Network Purely Memory Speedup (Optimized): {results['memory_speedup_optimized']:.2f}")
        print(f"Network Purely Memory Speedup (Arbitrary): {results['memory_speedup_arbitrary']:.2f}")
        print(f"Network Speedup (Optimized): {results['speedup_optimized']:.2f}")
        print(f"Network Speedup (Arbitrary): {results['speedup_arbitrary']:.2f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate neural network performance with different bitlengths and hardware models.")
    parser.add_argument('--data', type=str, default=None, help='Path to the folder containing the network\'s CSV descriptor and the bitlengths.')
    parser.add_argument('--hardware-json', type=str, default=None, help='Path to the JSON file containing the hardware model.')
    parser.add_argument('--baseline-bitlength', type=str, default="FP32", help='Bitlength used by baseline with which to compare the method against.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size of inputs that will be simulated on the network.')
    parser.add_argument('--software-overhead', type=float, default=0.0, help='Overhead that the method has when compressing/quantizing the values of the network (0.0 - 1.0).')
    parser.add_argument('--script-run', action='store_true', default=False, help='Less verbose version that only outputs the total speedups for both available and arbitrary bitlengths.')

    args = parser.parse_args()
    main(args)