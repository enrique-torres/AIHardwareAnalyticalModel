import csv
import json
import os

class Layer:
    def __init__(self, name, next_layer, layer_type, num_act_in_gradients, num_act_out_gradients, num_weight_gradients, forward_macs, extra_compute, batch_size, datatype):
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
        self.datatype = datatype

def read_network_metadata(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def parse_parameters_and_create_layers(args, file_name = "network_parameters.csv"):
    print(args.data)
    network_metadata = read_network_metadata(os.path.join(args.data, "network_metadata.json"))
    print(os.path.join("network_descriptions", network_metadata['network_name']))
    print(os.path.join("network_descriptions", network_metadata['network_name'], file_name))
    layers_dict = {}
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

            new_layer = Layer(layer_name, next_layer, layer_type, num_act_in_gradients, num_act_out_gradients, num_weight_gradients, forward_macs, extra_compute, args.batch_size, network_metadata["datatype"])
            layers_dict[layer_name] = new_layer
    return layers_dict