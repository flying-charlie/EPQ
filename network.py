import random
import math
from typing import Self
from collections.abc import Callable

# TODO:
# Write Docstrings for all classes/functions (only long ones for the Network class and subfunctions thereof) (Later?)

# region Standard functions

def Sigmoid(inputs: list[float], weights: list[float]):
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
            raise ValueError(f"The length of weights ({len(inputs)}) must be equal to the number of nodes in prev_layer ({self.getNodeCount()})")
        
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
    prev_layer = None
    value = None
    activation_function = None


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

        if len(weights) != prev_layer.getNodeCount(): # make sure there are the correct number of weights
            raise ValueError(f"The length of weights ({len(weights)}) must be equal to the number of nodes in prev_layer ({prev_layer.getNodeCount()})")
        
        self.weights = weights
        self.prev_layer = prev_layer
        self.activation_function = activation_function
    
    def activate(self):
        self.value = self.activation_function(self.prev_layer.getValues(), self.weights)


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
    '''
    input_layer : Layer = None
    hidden_layers : list[Layer] = []
    output_layer : Layer = None
    # activation_function_f : Callable[[list[float], list[float]], float] # A function taking a list of input signals and a list of weights and returning an output value for a node

    def __init__(self) -> None:
        self.hidden_layers = []

    def createFromWeights(weights: list[list[list[float]]], 
                          activation_function: Callable[[list[float], list[float]], float] = Sigmoid) -> Self: 
        
        self = Network.createEmpty()

        input_layer_size = len(weights[0][0]) # size of input layer = number of weights for each node in the first hidden layer
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

        weights = []

        for layer_size in hidden_layer_sizes:
            if not weights: # if weights is empty
                weights += [[[weight_initialisation_function(input_layer_size) for _ in range(input_layer_size)] for _ in range(layer_size)]] # create a layer of weights using the input layer size for the number of weights
            else:
                weights += [[[weight_initialisation_function(len(weights[-1])) for _ in range(len(weights[-1]))] for _ in range(layer_size)]] # else create a layer of weights based on the size of the previous layer

        weights += [[[weight_initialisation_function(len(weights[-1])) for _ in range(len(weights[-1]))] for _ in range(output_layer_size)]] # add the output layer weights

        return Network.createFromWeights(weights, activation_function=activation_function)

    def createEmpty() -> Self:
        
        self = Network()

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
    
    def activate(self, inputs: list[float]) -> list[float]:
        self.input_layer.setInputs(inputs)
        
        for layer in self.hidden_layers:
            layer.activate()
        self.output_layer.activate()

        return self.output_layer.getOutputs()