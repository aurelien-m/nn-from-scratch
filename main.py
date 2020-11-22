import numpy as np
import matplotlib.pyplot as plt

def sigmoid(input):
    return 1 / (1 + np.exp(-input))

def sigmoid_derivate(input):
    return sigmoid(input) * (1 - sigmoid(input))

# define dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 1]).reshape(4, 1)

weights = np.array([[0.1], [0.2]])
bias = 0.3
learning_rate = 0.1

errors = []

for epoch in range(1000):
    # feedforward
    in_feed = np.dot(inputs, weights) + bias
    out_feed = sigmoid(in_feed)

    # backpropagation
    error = out_feed - labels
    errors.append(np.sum(error))

    # calculating derivative
    derror = error
    dout_feed = sigmoid_derivate(out_feed)

    deriv = derror * dout_feed
    deriv_final = np.dot(inputs.T, deriv)

    weights -= learning_rate * deriv_final

    # updating the bias weight value
    for i in deriv:
        bias -= learning_rate * i

plt.plot(range(1000), errors)
plt.xlabel('Iterations')
plt.ylabel('Error')
# plt.show()

print('Weights:', weights)
print('Bias:', bias)

test = np.array([1, 1])
result = sigmoid(np.dot(test, weights) + bias)
print(result)