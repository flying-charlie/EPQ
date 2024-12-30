from network import * 

class GradientDescentTrainer:
    network: Network
    iterations: int
    learning_rate: float
    stopping_threshold: float
    target_function: Callable[[list[float]], list[float]]
    

    def createWithRandomNetwork(self,
                 input_layer_size : int, 
                 hidden_layer_sizes : list[int], 
                 output_layer_size : int, 
                 iterations : int,
                 learning_rate : float,
                 stopping_threshold: float,
                 batch_size: int,
                 activation_function: Callable[[list[float], list[float]], float] = Sigmoid, 
                 weight_initialisation_function: Callable[[int], float] = XavierWeightInitialisation # input: n (num of nodes in prev layer), output: random initial edge weight
                 ):
        
        network = Network.createRandom(input_layer_size, hidden_layer_sizes, output_layer_size, activation_function, weight_initialisation_function)
        return GradientDescentTrainer.createFromNetwork(network, iterations, learning_rate, stopping_threshold, batch_size)

    
    def createFromNetwork(self,
                 network : Network, 
                 iterations : int,
                 learning_rate : float,
                 stopping_threshold: float,
                 batch_size: int,
                 ):
        trainer = GradientDescentTrainer()
        trainer.network = network
        trainer.iterations = iterations
        trainer.learning_rate = learning_rate
        trainer.stopping_threshold = stopping_threshold
        return trainer


    def meanSquaredError(self, a: list[float], b: list[float]):
        sum = 0
        
        for pair in zip(a, b):
            sum += (pair[0] - pair[1]) ** 2
        
        return sum / len(a)


    def backpropagate(self, input: list[float], target_output: list[float]):
        self.network.run(input)
        errors = [target - node.value for node, target in zip(layer.nodes, target_output)]
        next_adjustments = self.backpropagateLayer(self.network.output_layer, errors)    # TODO name this properly

        for layer in reversed(self.network.hidden_layers):
            next_adjustments = self.backpropagateLayer(layer, next_adjustments)

    
    def backpropagateLayer(self, layer, target_adjustments):
        next_adjustments = [0 for _ in layer.nodes[0].weights]

        for node, adjustment in zip(layer.nodes, target_adjustments):
            node.bias += adjustment * self.learning_rate

            for weight, connection, connection_adjustment in zip(node.weights, node.prev_layer.nodes, next_adjustments):
                connection_adjustment += weight * adjustment * self.learning_rate
                weight += connection.value * adjustment * self.learning_rate
        
        return next_adjustments


    def gradientDescent(self, input: list[float], target_output: list[float]):

        for _ in range(self.iterations):
            # Find the current cost
            cost = self.meanSquaredError(self.network.run(input), target_output)
            
            # check if change since previous iteration is below the stopping threshold
            if previous_cost and abs(previous_cost-cost) <= self.stopping_threshold:
                break

            previous_cost = cost