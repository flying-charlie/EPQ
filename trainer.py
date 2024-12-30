from network import * 

class GradientDescentTrainer:
    network: Network
    iterations: int
    learning_rate: float
    stopping_threshold: float
    target_function: Callable[[list[float]], list[float]]
    

    def createWithRandomNetwork(
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

    
    def createFromNetwork(
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
        errors = [target - node.value for node, target in zip(self.network.output_layer.nodes, target_output)]
        next_adjustments = self.backpropagateLayer(self.network.output_layer, errors)    # TODO name this properly

        for layer in reversed(self.network.hidden_layers):
            next_adjustments = self.backpropagateLayer(layer, next_adjustments)

    
    def backpropagateLayer(self, layer, target_adjustments):
        next_adjustments = [0 for _ in layer.nodes[0].weights]

        for node, adjustment in zip(layer.nodes, target_adjustments):
            node.bias += adjustment * self.learning_rate

            for i in range(len(node.weights)):
                next_adjustments[i] += node.weights[i] * adjustment * self.learning_rate
                node.weights[i] += node.prev_layer.nodes[i].value * adjustment * self.learning_rate
        
        return next_adjustments


    def gradientDescent(self, input: list[float], target_output: list[float]):

        for _ in range(self.iterations):
            # Find the current cost
            cost = self.meanSquaredError(self.network.run(input), target_output)
            
            # check if change since previous iteration is below the stopping threshold
            if previous_cost and abs(previous_cost-cost) <= self.stopping_threshold:
                break

            previous_cost = cost

'''
def f1(x):
    if x > 0:
        return 1
    else:
        return 0

def f2(x):
    return abs(math.sin(x))

def f3(x):
    return 

def g1(x):
    return x

def g2(x):
    if x < 0.5:
        return x
    else:
        return 0.5 - x

f = g1

trainer = GradientDescentTrainer.createWithRandomNetwork(1, [10] * 2, 1, 0, 0.02, 0, 1)
for i in range(100):
    expected = [f(0), f(0.5), f(1)]
    actual = [i[0] for i in [trainer.network.run([0]), trainer.network.run([0.5]), trainer.network.run([1])]]
    print(trainer.meanSquaredError(expected, actual))
    for _ in range(5000):
        rand = random.uniform(0, 1)
        trainer.backpropagate([rand], [f(rand)])
        

print("Finished:")
print(trainer.network.run([0]), f(0))
print(trainer.network.run([0.5]), f(0.5))
print(trainer.network.run([1]), f(1))

expected = [f(0), f(0.5), f(1)]
actual = [i[0] for i in [trainer.network.run([0]), trainer.network.run([0.5]), trainer.network.run([1])]]
print("error:", trainer.meanSquaredError(expected, actual))
'''