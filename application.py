from Matrix import Matrix
from NeuralNetwork import NeuralNetwork
import random


training_data = {
    (0, 0): [0],
    (1, 0): [1],
    (0, 1): [1],
    (1, 1): [0]
}

for j in range(100):
    nn = NeuralNetwork(2, 4, 1)
    a = 0
    for i in range(1000):
        inputs, target = random.choice(list(training_data.items()))
        nn.train(list(inputs), target)
        a += 1

    print("Input: (0,0), Target: 0, Output: " + "".join([str(x) for x in nn.feed_forward([0, 0])]))
    print("Input: (1,1), Target: 0, Output: " + "".join([str(x) for x in nn.feed_forward([1, 1])]))
    print("Input: (0,1), Target: 1, Output: " + "".join([str(x) for x in nn.feed_forward([0, 1])]))
    print("Input: (1,0), Target: 1, Output: " + "".join([str(x) for x in nn.feed_forward([1, 0])]))
    print("Trained " + str(a*j) + " times.")
    print()

string = 'hello'
print("".join([x for x in string if x == 'l']))

