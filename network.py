import random
import math
from typing import Self
from collections.abc import Callable

# TODO:
# Add node activation + running the network
# Write Docstrings for all classes/functions (only long ones for the Network class and subfunctions thereof) (Later?)

# region Standard functions

def Sigmoid():
    raise NotImplementedError # TODO

def XavierWeightInitialisation(n: int) -> float:
    '''
    Creates a random weight following the Xavier weight initialisation function

    Parameters
    ----------
    n : int
        The number of nodes in the previous layer
    '''

    min, max = -(1.0 / math.sqrt(n)), (1.0 / math.sqrt(n))
    return random.uniform(min, max)

# endregion

# region Layer types

class Layer:
    '''
    One layer of a neural network
    '''
    nodes = []
    prev_layer = None

    def __init__(self, weights, prev_layer) -> None:
        '''
        Initialise a new layer with set weights
        '''
        self.nodes = [Node(prev_layer, node_weights) for node_weights in weights]


class InputLayer(Layer):
    '''
    The input layer to a neural network
    '''
    def __init__(self, size) -> None:
        '''
        Initialise a new input layer with size nodes
        '''
        self.nodes = [InputNode() for _ in range(size)]


class OutputLayer(Layer):
    '''
    The output layer to a neural network
    '''

    def __init__(self, weights, prev_layer) -> None:
        '''
        Initialise a new layer with set weights
        '''
        self.nodes = [OutputNode(prev_layer, node_weights) for node_weights in weights]

# endregion

# region Node types

class Node:
    '''
    One node of a neural network
    '''
    weights = {} # a dictionary of nodes in the previous layer to weights
    value = None

    def __init__(self, prev_layer, weights) -> None:
        '''
        Initialise a node with given weights matching up with nodes from given previous layer

        Parameters
        ----------
        prev_layer : Layer
            The previous layer of the network.

        weights : list[float]
            A list of weights that matches up with each node in prev_layer. 
            Must be of equal length to prev_layer.
        '''

        if len(weights) != len(prev_layer.nodes): # make sure there are the correct number of weights
            raise ValueError(f"The length of weights ({len(weights)}) must be equal to the number of nodes in prev_layer ({len(prev_layer.nodes)})")
        
        self.weights = {prev_layer.nodes[i]: weights[i] for i in range(len(weights))} # create the dictionary of nodes to weights


class InputNode(Node):
    '''
    An input node
    '''

    def __init__(self) -> None:
        self.weights = None


class OutputNode(Node):
    '''
    An output node
    '''



# endregion

class Network:
    '''
    A neural network
    '''
    input_layer : Layer = None
    hidden_layers : list[Layer] = []
    output_layer : Layer = None
    activation_function_f : Callable[[list[float], list[float]], float] = None # A function taking a list of input signals and a list of weights and returning an output value for a node

    def createFromWeights(weights: list[list[list[float]]], 
                          activation_function: Callable[[list[float], list[float]], float] = Sigmoid) -> Self: 
        
        self = Network()

        self.activation_function_f = activation_function

        input_layer_size = len(weights[0][0]) # size of input layer = number of weights for each node in the first hidden layer
        self.input_layer = InputLayer(input_layer_size)

        for layer_weights in weights:
            if not self.hidden_layers: # if hidden_layers is empty
                self.hidden_layers += [Layer(layer_weights, self.input_layer)] # create a layer with the input layer as the previous layer
            else:
                self.hidden_layers += [Layer(layer_weights, self.hidden_layers[-1])] # else create a layer based on the previous layer
        
        del self.hidden_layers[-1]

        self.output_layer = OutputLayer(weights[-1], self.hidden_layers[-1])

        return self

    def createRandom(input_layer_size : int, 
                     hidden_layer_sizes : list[int], 
                     output_layer_size : int, 
                     activation_function: Callable[[list[float], list[float]], float] = Sigmoid, 
                     weight_initialisation_function: Callable[[int], float] = XavierWeightInitialisation # input: n (num of nodes in prev layer), output: random initial edge weight
                     ) -> Self: 

        weights = []

        for layer_size in hidden_layer_sizes:
            if not weights: # if weights is empty
                weights += [[[weight_initialisation_function(input_layer_size) for _ in range(input_layer_size)] for _ in range(layer_size)]] # create a layer of weights using the input layer size for the number of weights
            else:
                weights += [[[weight_initialisation_function(len(weights[-1])) for _ in range(len(weights[-1]))] for _ in range(layer_size)]] # else create a layer of weights based on the size of the previous layer

        weights += [[[weight_initialisation_function(len(weights[-1])) for _ in range(len(weights[-1]))] for _ in range(output_layer_size)]] # add the output layer weights

        return Network.createFromWeights(weights, activation_function=activation_function)

    def createEmpty(activation_function: Callable[[list[float], list[float]], float] = Sigmoid, ) -> Self:
        
        self = Network()

        self.activation_function_f = activation_function

        return self

    def initialiseLayers(self, input_layer_size : int, hidden_layer_sizes : list[int], output_layer_size : int) -> None:
        self.input_layer = Layer(input_layer_size)
        self.hidden_layers = [Layer(input_layer_size[len]) for len in hidden_layer_sizes]
        self.output_layer = Layer(output_layer_size)

    def randomiseWeights(self) -> None:
        '''
        Randomises the weights of all nodes in the network
        '''
        # TODO

# network = Network.createRandom(1, [1], 1)
# network1 = Network.createRandom(2, [3, 3], 2)