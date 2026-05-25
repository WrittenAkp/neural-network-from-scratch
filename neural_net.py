import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
outputs = np.array([[0], [1], [1], [0]])
np.random.seed(42)
weights1 = np.random.rand(2, 4)
weights2 = np.random.rand(4, 1)
learning_rate = 0.1
for epoch in range(10000):
    hidden = sigmoid(np.dot(inputs, weights1))
    prediction = sigmoid(np.dot(hidden, weights2))
    error = outputs - prediction
    d_output = error * sigmoid_derivative(prediction)
    error_hidden = d_output.dot(weights2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden)
    weights2 += hidden.T.dot(d_output) * learning_rate
    weights1 += inputs.T.dot(d_hidden) * learning_rate
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f'Epoch {epoch}, Loss: {loss:.4f}')
print("\nFinal Predictions:")
for i in range(len(inputs)):
    print(f"{inputs[i]} -> {prediction[i][0]:.4f} | expected: {outputs[i][0]}")