import random
import math
from typing import Self
from collections.abc import Callable

# TODO:
# Write Docstrings for all classes/functions (only long ones for the Network class and subfunctions thereof) (Later?)

# region Standard functions

def Sigmoid(inputs: list[float], weights: list[float], bias: float):
    s = sum([input * weight for input, weight in zip(inputs, weights)]) + bias
    return 1 / (1 + (math.e ** -s))

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

    def __init__(self, weights, prev_layer, activation_function) -> None:
        '''
        Initialise a new layer with set weights
        '''
        self.nodes = [Node(prev_layer, node_weights, activation_function) for node_weights in weights]
    
    def getNodeCount(self):
        return len(self.nodes)

    def activate(self):
        for node in self.nodes:
            node.activate()
    
    def getValues(self):
        return [node.value for node in self.nodes]


class InputLayer(Layer):
    '''
    The input layer to a neural network
    '''
    def __init__(self, size) -> None:
        '''
        Initialise a new input layer with size nodes
        '''
        self.nodes = [InputNode() for _ in range(size)]

    def setInputs(self, inputs: list[float]) -> None:
        if len(inputs) != self.getNodeCount():
            raise ValueError(f"The length of inputs ({len(inputs)}) must be equal to the number of nodes in the input layer ({self.getNodeCount()})")
        
        for node, value in zip(self.nodes, inputs):
            node.setValue(value)


class OutputLayer(Layer):
    '''
    The output layer to a neural network
    '''

    def __init__(self, weights, prev_layer, activation_function) -> None:
        '''
        Initialise a new layer with set weights
        '''
        self.nodes = [OutputNode(prev_layer, node_weights, activation_function) for node_weights in weights]
    
    def getOutputs(self):
        return [node.getValue() for node in self.nodes]

# endregion

# region Node types

class Node:
    '''
    One node of a neural network
    '''
    weights = None # a list of weights
    prev_layer: Layer = None
    value = None
    activation_function = None
    bias = None


    def __init__(self, prev_layer, weights, activation_function) -> None:
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

        if len(weights) != prev_layer.getNodeCount() + 1: # make sure there are the correct number of weights
            raise ValueError(f"The length of weights ({len(weights)}) must be equal to the number of nodes in prev_layer ({prev_layer.getNodeCount()}) + 1 (constant bias)")
        
        self.bias = weights[-1]
        self.weights = weights[:-1]
        self.prev_layer = prev_layer
        self.activation_function = activation_function
    
    def activate(self):
        self.value = self.activation_function(self.prev_layer.getValues(), self.weights, self.bias)


class InputNode(Node):
    '''
    An input node
    '''

    def __init__(self) -> None:
        self.weights = None
    
    def setValue(self, value):
        self.value = value


class OutputNode(Node):
    '''
    An output node
    '''

    def getValue(self):
        return self.value



# endregion

class Network:
    '''
    A neural network
    
    Methods
    -------
    createFromWeights(weights, activation_function)

    createRandom(input_layer_size, hidden_layer_sizes, output_layer_size, activation_function, weight_initialisation_function) 
        
    createEmpty()

    run(inputs)
    '''
    input_layer : Layer = None
    hidden_layers : list[Layer] = []
    output_layer : Layer = None

    def __init__(self) -> None:
        self.hidden_layers = []

    def createFromWeights(weights: list[list[list[float]]],
                          activation_function: Callable[[list[float], list[float]], float] = Sigmoid) -> Self: 
        
        '''
        Create a new neural network based on a set of specified weights

        Parameters
        ----------
        weights : list[list[list[float]]]
            The set of weights to create the network from.
            In the form of a 3 dimensional list with the hierarchy:

            - Layers
            - Nodes
            - Weights

            Each node must have a weight for each of the nodes in the previous layer + 1 constant term (the bias)

        activation_function : Callable[[list[float], list[float]], float], optional
            Default: Sigmoid

            The function used when activating node. Takes parameters:
            - inputs: list[float]
                The list of inputs taken from the previous layer of nodes
            - weights: list[float]
                The weights of the current node
            - bias: float
                The constant bias of the current node
        
        Returns
        -------
        network: Network
            The network created from the given weights
        '''

        self = Network.createEmpty()

        input_layer_size = len(weights[0][0]) - 1 # size of input layer = number of weights for each node in the first hidden layer
        self.input_layer = InputLayer(input_layer_size)

        for layer_weights in weights:
            if not self.hidden_layers: # if hidden_layers is empty
                self.hidden_layers += [Layer(layer_weights, self.input_layer, activation_function)] # create a layer with the input layer as the previous layer
            else:
                self.hidden_layers += [Layer(layer_weights, self.hidden_layers[-1], activation_function)] # else create a layer based on the previous layer
        
        del self.hidden_layers[-1]

        self.output_layer = OutputLayer(weights[-1], self.hidden_layers[-1], activation_function)

        return self

    def createRandom(input_layer_size : int, 
                     hidden_layer_sizes : list[int], 
                     output_layer_size : int, 
                     activation_function: Callable[[list[float], list[float]], float] = Sigmoid, 
                     weight_initialisation_function: Callable[[int], float] = XavierWeightInitialisation # input: n (num of nodes in prev layer), output: random initial edge weight
                     ) -> Self: 
        '''
        Create a new neural network with random weights, based on a set of specified layer sizes

        Parameters
        ----------
        input_layer_size : int
            The number of inputs to the neural network. Used to generate the input layer.
        
        hidden_layer_sizes : list[int]
            The sizes of the hidden layers of the network. Each value in the list will generate a hidden layer of that size.

        output_layer_size : int
            The number of outputs to the neural network. Used to generate the output layer.
        
        activation_function : Callable[[list[float], list[float]], float], optional
            Default: Sigmoid

            The function used when activating node. Takes parameters:
            - inputs: list[float]
                The list of inputs taken from the previous layer of nodes
            - weights: list[float]
                The weights of the current node
            - bias: float
                The constant bias of the current node

        weight_initialisation_function : Callable[[int], float], optional
            Default: XavierWeightInitialisation

            The function used when creating random weights. Takes parameters:
            - n: int
                The number of nodes in the previous layer of the network
        
        Returns
        -------
        network: Network
            The network created from the given layer sizes
        '''

        weights = []

        for layer_size in hidden_layer_sizes:
            if not weights: # if weights is empty
                weights += [[[weight_initialisation_function(input_layer_size) for _ in range(input_layer_size + 1)] for _ in range(layer_size)]] # create a layer of weights using the input layer size for the number of weights
            else:
                weights += [[[weight_initialisation_function(len(weights[-1])) for _ in range(len(weights[-1]) + 1)] for _ in range(layer_size)]] # else create a layer of weights based on the size of the previous layer

        weights += [[[weight_initialisation_function(len(weights[-1])) for _ in range(len(weights[-1]) + 1)] for _ in range(output_layer_size)]] # add the output layer weights

        return Network.createFromWeights(weights, activation_function=activation_function)

    def createEmpty() -> Self:
        """
        Create a new neural network with no layers

        Returns
        -------
        network: Network
            The created network
        """
        self = Network()

        return self
    
    def run(self, inputs: list[float]) -> list[float]:
        '''
        Runs the network, taking a set of inputs and returning a set of outputs

        Parameters
        ----------
        inputs : list[float]
            A list containing the inputs to the network. Must be the same length as the number of inputs of the network
        
        Returns
        -------
        outputs: list[float]
            A list containing the outputs from the network. Contains one float for each output from the network.
        '''
        self.input_layer.setInputs(inputs)
        
        for layer in self.hidden_layers:
            layer.activate()
        self.output_layer.activate()

        return self.output_layer.getOutputs()
