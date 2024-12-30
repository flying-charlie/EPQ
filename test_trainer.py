from trainer import *

def test_backpropagateLayer_makes_no_change_with_zero_adjustments():
    weights = [[-4, -3, -2], [-1, 0, 1], [2, 3, 4]]
    prev_layer = InputLayer(2)
    for node in prev_layer.nodes: node.value = 0
    layer = Layer(weights, prev_layer, Sigmoid)
    trainer = GradientDescentTrainer()
    trainer.learning_rate = 1

    assert trainer.backpropagateLayer(layer, [0, 0, 0]) == [0, 0]

    assert [node.weights + [node.bias] for node in layer.nodes] == weights