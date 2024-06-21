import csv
import json
import os
from typing import Dict, List

from .menu_helpers import *

class Layer:
    def __init__(self, name, next_layer, previous_layer, layer_type, num_act_in_gradients, num_act_out_gradients, num_weight_gradients, forward_macs, extra_compute, batch_size, network_metadata):
        self.name = name
        self.next_layer = next_layer
        self.previous_layer = previous_layer
        self.layer_type = layer_type
        self.num_act_in_gradients = num_act_in_gradients
        self.num_act_out_gradients = num_act_out_gradients
        self.num_weight_gradients = num_weight_gradients
        self.forward_macs = forward_macs
        self.extra_compute = extra_compute
        self.batch_size = batch_size
        self._activation_bitlengths = []
        self._weight_bitlengths = []
        self.network_metadata = network_metadata

def read_network_metadata(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def parse_parameters_and_create_layers(bitlengths_path, batch_size, file_name = "network_parameters.csv"):
    # Read the network name from the bitlengths to then read the network definition/parameters. Also read whether the
    # bitlengths that will be used for the simulation are integer bitlengths or floating point bitlengths
    network_metadata = read_network_metadata(os.path.join(bitlengths_path, "network_metadata.json"))
    #print(os.path.join("network_descriptions", network_metadata['network_name']))
    #print(os.path.join("network_descriptions", network_metadata['network_name'], file_name))
    layers_dict = {}
    previous_layer = None
    with open(os.path.join("network_descriptions", network_metadata['network_name'], file_name)) as csvfile:
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

            new_layer = Layer(layer_name, next_layer, previous_layer, layer_type, num_act_in_gradients, num_act_out_gradients, num_weight_gradients, forward_macs, extra_compute, batch_size, network_metadata)
            layers_dict[layer_name] = new_layer

            previous_layer = new_layer

    for layer_name in layers_dict.keys():
        if layers_dict[layer_name].next_layer != 'LAST_LAYER':
            next_layer = layers_dict[layers_dict[layer_name].next_layer]
            layers_dict[layer_name].next_layer = next_layer
        else:
            layers_dict[layer_name].next_layer = None
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

def get_closest_datatype(bitlength, datatypes, network_metadata):
    closest_type = None
    smallest_diff = float('inf')
    
    for dtype, specs in datatypes.items():
        if network_metadata["datatype"] == "integer":
            if "INT" in dtype:
                diff = specs['bitlength'] - bitlength
                if 0 <= diff < smallest_diff:
                    smallest_diff = diff
                    closest_type = dtype
        elif network_metadata["datatype"] == "float":
            if "FP" in dtype:
                diff = specs['bitlength'] - bitlength
                if 0 <= diff < smallest_diff:
                    smallest_diff = diff
                    closest_type = dtype
        else:
            datatype = network_metadata["datatype"]
            print(f"Unrecognized type of data selected for the bitlengths of the network: {datatype}. Exiting")
            exit(1)
    return closest_type

def load_network_and_bitlengths(bitlengths_path, batch_size):
    if bitlengths_path is None:
        bitlengths_path = select_directory("bitlength_data")
    if bitlengths_path is None:
        print("The data path argument needs to be provided in order to compute speedups. Exiting now.")
        exit(0)

    layers_dict = parse_parameters_and_create_layers(bitlengths_path, batch_size)
    parse_bitlengths(bitlengths_path, layers_dict)

    return layers_dict