import numpy as np

class MLP:
"""A Multilayer Perceptron class.
"""

    def __init__(self, num_inputs: int=3, num_hidden: list=[3, 5], num_outputs: int=2) -> None:
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs

        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initiate random weights
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

    def forward_propagate(self, inputs: list) -> list:

        activations = inputs

        for w in self.weights:
            # calculate net inputs
            net_inputs = np.dot(activations, w)

            # calculate the activations
            activations = self._sigmoid(net_inputs)

        return activations

    def _sigmoid(self, x: list) -> list:
        return 1.0 / (1+ np.exp(-x))


if __name__ == "__main__":
    # create a MLP
    mlp = MLP()

    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward propagation
    outputs = mlp.forward_propagate(inputs)

    # print results
    print("The network inputs are: {}".format(inputs))
    print("The network output is: {}".format(outputs))