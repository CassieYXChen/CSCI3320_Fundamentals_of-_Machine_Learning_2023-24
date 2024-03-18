import numpy as np
X = np.array([[4, 49],
              [5, 4],
              [12, 28],
              [29, 18],
              [30, 65],
              [36, 32],
              [36, 1],
              [54, 29],
              [58, 76],
              [70, 12],
              [72, 26],
              [76, 55],
              [78, 4],
              [82, 15],
              [87, 95],
              [90, 70],
              [90, 55],
              [92, 84],
              [95, 14],
              [98, 21]])
y = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
np.random.seed(42)
W = np.random.randn(X.shape[1], 1)
b = np.zeros((1, 1))

def sigmoid(z):
    clipped_z = np.clip(z, -500, 500)  # Clip the values of z to avoid overflow
    return 1 / (1 + np.exp(-clipped_z))

def forward_propagation(X, W, b):
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    return A

def backward_propagation(X, A, y):
    dZ = A - y.reshape(-1, 1)
    dW = np.dot(X.T, dZ)
    db = np.sum(dZ, axis=0, keepdims=True)
    return dW, db

def gradient_descent(X, y, W, b, learning_rate, num_iterations):
    for i in range(num_iterations):
        A = forward_propagation(X, W, b)
        dW, db = backward_propagation(X, A, y)
        W -= learning_rate * dW
        b -= learning_rate * db
    return W, b

learning_rate = 0.01
num_iterations = 1000
W, b = gradient_descent(X, y, W, b, learning_rate, num_iterations)
# Calculate accuracy:
y_pred = forward_propagation(X, W, b)
y_pred_class = np.round(y_pred)
accuracy = np.mean(y_pred_class == y.reshape(-1, 1)) * 100
print(f"Accuracy on the training dataset: {accuracy}%")