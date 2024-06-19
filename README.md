# AI Hardware Analytical Model
A modular hardware analytical model to calculate estimated speedups from memory compression and quantization techniques for AI, on different hardware architectures. 

The main objective of this model is to quickly iterate over different hardware architectures and estimate speedups for memory compression and quantization techniques being researched for AI.
To run the analytical model you will need to execute the following command:

***python inference_analytical_model.py***

Optional arguments:
  - --data (Path to the folder containing the network\'s CSV descriptor and the bitlengths.)
  - --hardware-json (Path to the JSON file containing the hardware model.)
  - --baseline-bitlength (Bitlength used by baseline with which to compare the method against.)
  - --batch-size (Batch size of inputs that will be simulated on the network.)
  - --software-overhead (Overhead that the method has when compressing/quantizing the values of the network (0.0 - 1.0).)

If ***--data*** or ***--hardware-json*** are left empty, a list of selections will be provided for both to the user running the analytical model. These selections will be read from folders that shall be placed on the root directory of the script. 

For the ***--data*** argument, the different options will be read from a folder called bitlength_data. This folder will have to contain a sub-folder per network that needs to be simulated. Inside each network sub-folder, there need to be three files: parameter_values.csv, activation_bitlengths.csv and weights_bitlengths.csv. These files contain the network definition, with the per layer number of parameters, activations and operations; the compressed/quantized bitlengths per layer for the activations of the network, with the first row of the CSV file requiring the ordered list of all the layers' names (as declared on the parameter_values.csv file); and the compressed/quantized bitlengths per layer for the weights of the network, with the same format as the activations; respectively.

For the ***--hardware-json*** argument, the options will be gathered from the names of the .json files contained in a folder called hardware_descriptions. Each .json file must accurately describe the to be simulated hardware, with memory bandwiths and latencies for the different memory levels in the hierarchy, and with the different throughputs for the different supported datatypes by the architecture. See the included .json files for reference.
