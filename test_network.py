from network import *
import pytest
from mock import patch

class Helper:
    def get_node_weights(node: Node) -> list[float]:
        return list(node.weights) + [node.bias]
    
    def get_network_weights(network: Network) -> list[list[list[float]]]:
        weights = []
        for layer in network.hidden_layers:
            weights.append([Helper.get_node_weights(node) for node in layer.nodes])
        weights.append([Helper.get_node_weights(node) for node in network.output_layer.nodes])
        return weights


# region Node Tests

def test_node_init_creates_valid_empty_node():
    prev_layer = InputLayer(0)
    node = Node(prev_layer, [0], None)
    assert Helper.get_node_weights(node) == [0]

def test_node_init_creates_valid_node_from_weights():
    prev_layer = InputLayer(3)
    weights = [0, 5, 0.3, 0]
    node = Node(prev_layer, weights, None)
    assert Helper.get_node_weights(node) == weights

def test_node_init_errors_when_passed_incorrect_num_of_weights():
    prev_layer = InputLayer(3)
    weights = [0, 5, 0]
    with pytest.raises(ValueError):
        Node(prev_layer, weights, None)

    weights = [0, 5, 7, 0.4, 0]
    with pytest.raises(ValueError):
        Node(prev_layer, weights, None)

# endregion


# region Layer Tests

def test_input_layer_init_creates_valid_empty_input_layer():
    layer = InputLayer(0)
    assert not layer.nodes

def test_input_layer_init_creates_input_layer_with_correct_number_of_nodes():
    layer = InputLayer(3)
    assert len(layer.nodes) == 3

def test_input_layer_init_creates_input_layer_of_nodes_with_no_weights():
    layer = InputLayer(3)
    assert layer.nodes[0].weights == None

def test_layer_init_creates_valid_empty_layer():
    prev_layer = InputLayer(3)
    layer = Layer([], prev_layer, None)
    assert not layer.nodes

def test_layer_init_creates_layer_with_correct_number_of_nodes():
    prev_layer = InputLayer(2)
    layer = Layer([[0, 0, 0], [0, 0, 0], [0, 0, 0]], prev_layer, None)
    assert len(layer.nodes) == 3

def test_layer_init_creates_layer_with_correct_number_of_weights_per_node():
    prev_layer = InputLayer(2)
    layer = Layer([[0, 0, 0], [0, 0, 0], [0, 0, 0]], prev_layer, None)
    assert len(layer.nodes[0].weights) == 2

def test_output_layer_init_creates_output_layer_with_correct_number_of_nodes():
    prev_layer = InputLayer(2)
    layer = OutputLayer([[0, 0, 0], [0, 0, 0], [0, 0, 0]], prev_layer, None)
    assert len(layer.nodes) == 3

def test_output_layer_init_creates_output_layer_with_correct_number_of_weights_per_node():
    prev_layer = InputLayer(2)
    layer = OutputLayer([[0, 0, 0], [0, 0, 0], [0, 0, 0]], prev_layer, None)
    assert len(layer.nodes[0].weights) == 2

# endregion


# region Network Tests

def test_network_createEmpty_creates_valid_empty_network():
    network: Network = Network.createEmpty()
    assert network.hidden_layers == []
    assert network.input_layer == None
    assert network.output_layer == None

def test_network_createFromWeights_creates_correct_network():
    weights = [
        [[0, 1, 4], [2, 3, 3], [4, 5, 1]],
        [[6, 7, 8, 9], [9, 10, 11, 12]]
    ]
    network: Network = Network.createFromWeights(weights)
    assert Helper.get_network_weights(network) == weights

def test_network_createFromWeights_creates_valid_network_with_alternative_activation_function():
    def activation_function():
        pass

    weights = [
        [[0, 1, 4], [2, 3, 3], [4, 5, 1]],
        [[6, 7, 8, 9], [9, 10, 11, 12]]
    ]
    network: Network = Network.createFromWeights(weights, activation_function)

    assert network.hidden_layers[0].nodes[0].activation_function == activation_function
    assert network.output_layer.nodes[0].activation_function == activation_function

def test_network_createRandom_creates_network_with_valid_node_distribution():

    network = Network.createRandom(2, [3, 3], 2)
    
    expected_zero_weights = [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0]]
    ]
    
    zero_weights = [[[0 for weight in node] for node in layer] for layer in Helper.get_network_weights(network)]

    assert zero_weights == expected_zero_weights

def test_network_createRandom_creates_correct_network_for_monkeypatched_random_function():
    def zero(*_):
        return 0.0
    
    with patch('random.uniform', new = lambda *_: 0.0):
        network = Network.createRandom(2, [3, 3], 2)
    
    expected_weights = [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0]]
    ]
    
    assert Helper.get_network_weights(network) == expected_weights

def test_network_createRandom_creates_correct_network_for_alternative_weight_initialisation_function():

    network = Network.createRandom(2, [3, 3], 2, weight_initialisation_function = lambda *_: 0.0)
    
    expected_weights = [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0]]
    ]
    
    assert Helper.get_network_weights(network) == expected_weights

def test_network_activation_gives_correct_output_for_sum_activation_function():
    def activation(inputs: list[float], weights: list[float], bias: float):
        return sum(inputs)
    network = Network.createRandom(2, [3, 3], 2, activation)
    assert network.activate([1, 1]) == [18, 18]

def test_network_activation_gives_correct_output_for_default_activation_function():
    weights = [
        [[1, 1, 0]],
        [[0.5, 0], [6, 0]]
    ]
    network = Network.createFromWeights(weights)
    assert network.activate([1, 1]) == [0.6083539905113917974904406486799764342316532211631442627440961721, 0.9949574219138469816618885150449085292094590358722334752380849723] # calculated using https://www.wolframalpha.com/
# endregion