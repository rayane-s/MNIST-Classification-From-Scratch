import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load Dataset
mnist_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

mnist_data = np.array(mnist_data)
total_samples, num_features = mnist_data.shape
np.random.shuffle(mnist_data)

# Split into dev and training sets
dev_data = mnist_data[0:1000].T
dev_labels = dev_data[0]
dev_features = dev_data[1:num_features]
dev_features = dev_features / 255.

training_data = mnist_data[1000:total_samples].T
training_labels = training_data[0]
training_features = training_data[1:num_features]
training_features = training_features / 255.
_, num_train_samples = training_features.shape

def initialize_network_params():
    layer1_weights = np.random.rand(10, 784) - 0.5
    layer1_bias = np.random.rand(10, 1) - 0.5
    layer2_weights = np.random.rand(10, 10) - 0.5
    layer2_bias = np.random.rand(10, 1) - 0.5
    return layer1_weights, layer1_bias, layer2_weights, layer2_bias

def relu_activation(Z):
    return np.maximum(Z, 0)

def softmax_activation(Z):
    exps = np.exp(Z)
    return exps / np.sum(exps)

def forward_propagation(layer1_weights, layer1_bias, layer2_weights, layer2_bias, input_features):
    z1 = layer1_weights.dot(input_features) + layer1_bias
    a1 = relu_activation(z1)
    z2 = layer2_weights.dot(a1) + layer2_bias
    a2 = softmax_activation(z2)
    return z1, a1, z2, a2

def relu_derivative(Z):
    return Z > 0

def one_hot_encode(labels):
    num_classes = labels.max() + 1
    one_hot_matrix = np.zeros((labels.size, num_classes))
    one_hot_matrix[np.arange(labels.size), labels] = 1
    return one_hot_matrix.T

def backward_propagation(z1, a1, z2, a2, layer1_weights, layer2_weights, input_features, labels):
    one_hot_labels = one_hot_encode(labels)
    dz2 = a2 - one_hot_labels
    dw2 = 1 / num_train_samples * dz2.dot(a1.T)
    db2 = 1 / num_train_samples * np.sum(dz2)
    dz1 = layer2_weights.T.dot(dz2) * relu_derivative(z1)
    dw1 = 1 / num_train_samples * dz1.dot(input_features.T)
    db1 = 1 / num_train_samples * np.sum(dz1)
    return dw1, db1, dw2, db2

def update_network_params(layer1_weights, layer1_bias, layer2_weights, layer2_bias, 
                           dw1, db1, dw2, db2, learning_rate):
    layer1_weights -= learning_rate * dw1
    layer1_bias -= learning_rate * db1    
    layer2_weights -= learning_rate * dw2  
    layer2_bias -= learning_rate * db2    
    return layer1_weights, layer1_bias, layer2_weights, layer2_bias

def get_model_predictions(a2):
    return np.argmax(a2, 0)

def calculate_accuracy(predictions, true_labels):
    print("Predictions:", predictions)
    print("True Labels:", true_labels)
    return np.sum(predictions == true_labels) / true_labels.size

def train_neural_network(input_features, labels, learning_rate, num_iterations):
    layer1_weights, layer1_bias, layer2_weights, layer2_bias = initialize_network_params()
    
    for iteration in range(num_iterations):
        z1, a1, z2, a2 = forward_propagation(layer1_weights, layer1_bias, layer2_weights, layer2_bias, input_features)
        dw1, db1, dw2, db2 = backward_propagation(z1, a1, z2, a2, layer1_weights, layer2_weights, input_features, labels)
        layer1_weights, layer1_bias, layer2_weights, layer2_bias = update_network_params(
            layer1_weights, layer1_bias, layer2_weights, layer2_bias, 
            dw1, db1, dw2, db2, learning_rate
        )
        
        if iteration % 10 == 0:
            print(f"Iteration: {iteration}")
            predictions = get_model_predictions(a2)
            print("Accuracy:", calculate_accuracy(predictions, labels))
    
    return layer1_weights, layer1_bias, layer2_weights, layer2_bias

def predict_single_image(input_image, layer1_weights, layer1_bias, layer2_weights, layer2_bias):
    _, _, _, a2 = forward_propagation(layer1_weights, layer1_bias, layer2_weights, layer2_bias, input_image)
    return get_model_predictions(a2)

def visualize_prediction(index, features, labels, layer1_weights, layer1_bias, layer2_weights, layer2_bias):
    current_image = features[:, index, None]
    prediction = predict_single_image(current_image, layer1_weights, layer1_bias, layer2_weights, layer2_bias)
    label = labels[index]
    
    print("Prediction:", prediction[0])
    print("Label:", label)
    
    current_image_display = current_image.reshape((28, 28)) * 255
    plt.figure(figsize=(5,5))
    plt.gray()
    plt.imshow(current_image_display, interpolation='nearest')
    plt.title(f"Predicted: {prediction[0]}, Actual: {label}")
    plt.axis('off')
    plt.show()

# Train the neural network
print("Starting Neural Network Training...")
trained_w1, trained_b1, trained_w2, trained_b2 = train_neural_network(
    training_features, training_labels, learning_rate=0.10, num_iterations=500
)

# Visualize a few predictions
print("\nVisualization of some predictions:")
for test_index in range(5):
    visualize_prediction(
        test_index, 
        training_features, 
        training_labels, 
        trained_w1, trained_b1, 
        trained_w2, trained_b2
    )
