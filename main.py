from random import randint
from MLP import MLP
from numpy import array

mlp = MLP()

rInput = randint(0, 1)
inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_data = array([[0, 1, 1, 0]]).T

mlp.train_weights(training_data, inputs)

new_input = array((input('X:'), input('Y:'), input('Z:')), dtype=int)
new_input_result = mlp.feedforward(new_input)

print()
print("New input values: ")
print(new_input)
print()
print("New input result: ")
print(new_input_result)