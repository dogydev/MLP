from numpy import random, dot, exp


class MLP(object):

    def __init__(self):
        self.weights = 2 * random.random((3, 1)) - 1

        self.lr = 0.003

        print("Random Weights: ")
        print(self.weights)

    def train_weights(self, target, inputs):
        for i in range(len(target) * 100000):
            result = self.feedforward(inputs)
            error = target - result

            print()
            print("Error: ")
            print(error)
            print()

            weights_adjustment = dot(inputs.T, error * self.lr * self.sigmoid_deriv(result))

            self.weights += weights_adjustment

            print("New Weights: ")
            print(self.weights)

    def feedforward(self, inputs):
        return self.sigmoid_function(dot(inputs, self.weights))

    def sigmoid_function(self, x):
        return 1 / (1 + exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)
