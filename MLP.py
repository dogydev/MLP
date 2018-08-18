"""MIT License

Copyright (c) 2018 dogydev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

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
