import wandb
import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist

# load the fashion-mnist dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# class labels
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# one sample per class
sample_images = []
sample_labels = []
for class_idx in range(10):
    index = np.where(y_train == class_idx)[0][0]  # get the first occurence
    sample_images.append(x_train[index])
    sample_labels.append(class_names[class_idx])

# initialize wandb
wandb.init(project="fashion-mnist-classification",
           entity="tejaswiniksssn-indian-institute-of-technology-madras",
           name="fashion-mnist-samples")
# log images into wandb
wandb.log({"fashion-mnist samples": [wandb.Image(img, caption=label) for img, label in zip(sample_images, sample_labels)]})
# finish wandb run
wandb.finish()

# ---------------------------
# Activation Functions & Derivatives
# ---------------------------
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x)**2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    # Subtract max for numerical stability.
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Activation lookup dictionary
ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "softmax": (softmax, None)  # Softmax derivative is computed with cross-entropy.
}


def activate(x, act_type):
    if act_type in ACTIVATIONS:
        return ACTIVATIONS[act_type][0](x)
    else:
        raise ValueError("Unsupported activation type: " + str(act_type))


def activation_derivative(x, act_type):
    if act_type in ACTIVATIONS:
        deriv = ACTIVATIONS[act_type][1]
        if deriv is None:
            raise ValueError("Derivative for activation type '{}' not implemented.".format(act_type))
        return deriv(x)
    else:
        raise ValueError("Unsupported activation type: " + str(act_type))


def compute_accuracy(y_true, y_pred):
        correct = (y_true == y_pred).sum()
        total = len(y_true)
        return correct / total


   
# ---------------------------
# Neural Network Class
# ---------------------------
class FeedForwardNN:
    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_layer_dim, lr=0.01,
                 act_hidden="sigmoid", act_output="softmax", weight_init="random", weight_decay = 0.0):
        """
        Initializes the feed-forward neural network.
        Bias is incorporated by appending a column of ones to the input of each layer.
        """
        self.lr = lr
        self.act_hidden = act_hidden
        self.act_output = act_output
        self.weight_init = weight_init
        self.num_hidden_layers = num_hidden_layers


        # Define layer dimensions: input, hidden layers, output.
        self.layer_dims = [input_dim] + [hidden_layer_dim] * num_hidden_layers + [output_dim]
        self.num_layers = len(self.layer_dims) - 1  # total number of weight matrices


        # Initialize weights (with bias included) using He initialization.
        self.W = self.initialize_weights(input_dim, hidden_layer_dim, output_dim)


    def initialize_weights(self, input_dim, hidden_dim, output_dim):
        """Initialize weights using the selected method (random or Xavier)."""
        W = []
        layer_sizes = [input_dim] + [hidden_dim] * self.num_hidden_layers + [output_dim]


        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i] + 1  # +1 for bias term
            output_size = layer_sizes[i + 1]
       
            if self.weight_init == "xavier":
                limit = np.sqrt(6 / (input_size + output_size))
                W.append(np.random.uniform(-limit, limit, (input_size, output_size)))
            else:  # Default to random normal
                W.append(np.random.randn(input_size, output_size) * 0.01)
        return W


    def forward(self, X):
        """
        Forward propagation.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_dim)
        Returns:
            np.ndarray: Output probabilities of shape (n_samples, output_dim)
        """
        n_samples = X.shape[0]
        self.a_list = []  # activations (including bias)
        self.z_list = []  # pre-activations


        # Append bias term to input.
        a = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        self.a_list.append(a)


        for i in range(self.num_layers):
            z = a @ self.W[i]
            self.z_list.append(z)
            if i < self.num_layers - 1:
                a = activate(z, self.act_hidden)
                # Append bias term for next layer.
                a = np.concatenate([a, np.ones((n_samples, 1))], axis=1)
            else:
                # For output layer, use softmax activation.
                a = activate(z, self.act_output)
            self.a_list.append(a)


        return a


    def compute_loss(self, Y_pred, Y_true, weight_decay=0.0):
        """
        Computes cross-entropy loss for multi-class classification.
        Args:
            Y_pred (np.ndarray): Predicted probabilities (n_samples, output_dim)
            Y_true (np.ndarray): One-hot encoded true labels (n_samples, output_dim)
        Returns:
            float: Cross-entropy loss.
        """
        """Compute loss with optional weight decay."""
        loss = np.mean(-np.sum(Y_true * np.log(Y_pred + 1e-8), axis=1))  # Cross-entropy loss
   
        # Add weight decay (L2 Regularization)
        if weight_decay > 0:
            l2_penalty = sum(np.sum(W ** 2) for W in self.W) * (weight_decay / 2)
            loss += l2_penalty
        return loss
   


    def backward(self, X, Y_true, weight_decay = 0.0):
        """
        Backward propagation: computes gradients for weights.
        Args:
            X (np.ndarray): Input data (n_samples, input_dim)
            Y_true (np.ndarray): One-hot encoded true labels (n_samples, output_dim)
        Returns:
            list: Gradients for each weight matrix.
        """
        grads = [None] * self.num_layers
        n_samples = X.shape[0]
        Y_pred = self.a_list[-1]


        # Backpropagate through layers.
        for i in reversed(range(self.num_layers)):
            z = self.z_list[i]
            a_prev = self.a_list[i]  # activation (with bias) from previous layer


            if i == self.num_layers - 1:
                # For softmax with cross-entropy loss.
                dZ = (Y_pred - Y_true) / n_samples
            else:
                dZ = dA * activation_derivative(z, self.act_hidden)


            # Compute gradient for weight matrix.
            dW = a_prev.T @ dZ
            if weight_decay > 0:
                dW += weight_decay * self.W[i]
            grads[i] = dW


            # Propagate gradient to previous layer (exclude bias column).
            dA = dZ @ self.W[i].T
            dA = dA[:, :-1]


        return grads


    def update_params(self, grads):
        """
        Updates weights using simple SGD.
        """
        for i in range(self.num_layers):
            self.W[i] -= self.lr * grads[i]


    def train(self, X, Y, epochs=20, batch_size=None, print_every=1, optimizer=None):
        """
        Trains the neural network with support for mini-batch processing.
        Args:
            X (np.ndarray): Input data (n_samples, input_dim)
            Y (np.ndarray): One-hot encoded labels (n_samples, output_dim)
            epochs (int): Number of training epochs.
            batch_size (int): Mini-batch size. If None, full batch gradient descent is used.
            print_every (int): Frequency of printing loss.
            optimizer: Optional optimizer object.
        Returns:
            list: Loss history over epochs.
        """
        n_samples = X.shape[0]
        # If batch_size is not provided, use full batch.
        if batch_size is None:
            batch_size = n_samples


        loss_history = []
        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch.
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]


            epoch_loss = 0.0
            num_batches = int(np.ceil(n_samples / batch_size))


            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]


                # Forward pass for mini-batch.
                Y_pred = self.forward(X_batch)
                loss = self.compute_loss(Y_pred, Y_batch)
                epoch_loss += loss * (X_batch.shape[0] / n_samples)


                # Backward pass and update.
                grads = self.backward(X_batch, Y_batch)
                if optimizer is not None:
                    optimizer.step(grads)
                else:
                    self.update_params(grads)


            loss_history.append(epoch_loss)
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")


        return loss_history


# ---------------------------
# Optimizer Implementation
# ---------------------------
class Optimizer:
    def __init__(self, parameters, optimizer_type="sgd", lr=0.01, momentum=0.9, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes the optimizer.
        Args:
            parameters (list): List of parameters (weights) to optimize.
            optimizer_type (str): One of 'sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam'.
            lr (float): Learning rate.
            momentum (float): Momentum factor (for momentum/nesterov).
            beta (float) : factor for rmsprop
            beta1 (float): Decay rate for first moment estimates (adam/nadam).
            beta2 (float): Decay rate for second moment estimates (adam/nadam).
            epsilon (float): Small constant to prevent division by zero.
        """
        self.parameters = parameters
        self.optimizer_type = optimizer_type.lower()
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # time step for Adam/Nadam


        if self.optimizer_type in ["momentum", "nesterov"]:
            self.velocity = [np.zeros_like(p) for p in parameters]
        if self.optimizer_type == "rmsprop":
            self.cache = [np.zeros_like(p) for p in parameters]
        if self.optimizer_type in ["adam", "nadam"]:
            self.m = [np.zeros_like(p) for p in parameters]
            self.v = [np.zeros_like(p) for p in parameters]


    def step(self, grads):
        if self.optimizer_type == "sgd":
            for i in range(len(self.parameters)):
                self.parameters[i] -= self.lr * grads[i]
        elif self.optimizer_type == "momentum":
            for i in range(len(self.parameters)):
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
                self.parameters[i] += self.velocity[i]
        elif self.optimizer_type == "nesterov":
            for i in range(len(self.parameters)):
                v_prev = self.velocity[i].copy()
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
                self.parameters[i] += -self.momentum * v_prev + (1 + self.momentum) * self.velocity[i]
        elif self.optimizer_type == "rmsprop":
            for i in range(len(self.parameters)):
                self.cache[i] = self.beta * self.cache[i] + (1-self.beta) * (grads[i] ** 2)
                self.parameters[i] -= self.lr * grads[i] / (np.sqrt(self.cache[i]) + self.epsilon)
        elif self.optimizer_type == "adam":
            self.t += 1
            for i in range(len(self.parameters)):
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                self.parameters[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        elif self.optimizer_type == "nadam":
            self.t += 1
            for i in range(len(self.parameters)):
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                nesterov_term = (self.beta1 * m_hat + (1 - self.beta1) * grads[i] / (1 - self.beta1 ** self.t))
                self.parameters[i] -= self.lr * nesterov_term / (np.sqrt(v_hat) + self.epsilon)
        else:
            raise ValueError("Unsupported optimizer type: " + self.optimizer_type)
       
# ---------------------------
# Main: Train on Fashion MNIST (10 classes) with Mini-Batch Training
# ---------------------------
if __name__ == '__main__':
    # Load Fashion MNIST dataset.
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


    # Preprocess: reshape images to vectors and scale pixel values to [0, 1].
    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0


    # One-hot encode the labels.
    num_classes = 10
    Y_train = np.eye(num_classes)[y_train]
    Y_test = np.eye(num_classes)[y_test]


    # Neural network configuration.
    input_dim = 28 * 28  # 784 features.
    output_dim = num_classes  # 10 classes.
    num_hidden_layers = 2
    hidden_layer_dim = 128
    learning_rate = 0.01
    epochs = 20  # You may increase epochs for better performance.
    batch_size = 128  # Mini-batch size.
    weight_init = "random"
    weight_decay = 0.0


    # Initialize the neural network.
    nn = FeedForwardNN(input_dim, output_dim, num_hidden_layers, hidden_layer_dim,
                       lr=learning_rate, act_hidden="sigmoid", act_output="softmax", weight_init = "random", weight_decay = 0.0)


    # Select optimizer (options: "sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam").
    optimizer_type = "adam"
    optimizer = Optimizer(parameters=nn.W, optimizer_type=optimizer_type, lr=learning_rate)


    # Train the network with mini-batch training.
    print("Training started with mini-batch training...\n")
    loss_history = nn.train(x_train, Y_train, epochs=epochs, batch_size=batch_size, print_every=1, optimizer=optimizer)


    # Evaluate on training data.
    Y_pred_train = nn.forward(x_train)
    train_predictions = np.argmax(Y_pred_train, axis=1)
    train_accuracy = compute_accuracy(y_train, train_predictions)
    print("\nTraining Accuracy: {:.2f}%".format(train_accuracy * 100))


   


# Define the sweep configuration
sweep_config = {
    "method": "random",  # Random search strategy for efficiency
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [5, 10]},
        "num_hidden_layers": {"values": [3, 4, 5]},
        "hidden_layer_size": {"values": [32, 64, 128]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]}
    }
}


# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-hyperparam-tuning")


# Load and preprocess data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# split training data into train & validation (90%-10%)
val_split = int(0.9 * len(x_train))
x_train, x_val = x_train[:val_split], x_train[val_split:]
y_train, y_val = y_train[:val_split], y_train[val_split:]


x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_val = x_val.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0


num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_val = np.eye(num_classes)[y_val]
y_test = np.eye(num_classes)[y_test]


def train(config=None):
    run = wandb.init(
        project="fashion-mnist-sweep",
        entity="tejaswiniksssn-indian-institute-of-technology-madras",
        config=config
    )


    config = wandb.config  # Get hyperparameters from sweep


    # Set a proper run name after config is initialized
    run.name = f"hl_{config.num_hidden_layers}_bs_{config.batch_size}_ac_{config.activation}"


    # Initialize the neural network
    nn = FeedForwardNN(
        input_dim=28 * 28,
        output_dim=10,
        num_hidden_layers=config.num_hidden_layers,
        hidden_layer_dim=config.hidden_layer_size,
        lr=config.learning_rate,
        act_hidden=config.activation,
        act_output="softmax",
        weight_init=config.weight_init,
        weight_decay=config.weight_decay
    )


    optimizer = Optimizer(parameters=nn.W, optimizer_type=config.optimizer, lr=config.learning_rate)
    best_val_accuracy = 0
    for epoch in range(config.epochs):
        # Train the model and record loss
        loss = nn.train(x_train, y_train, epochs=1, batch_size=config.batch_size, optimizer=optimizer)
        # Ensure loss is a scalar
        if isinstance(loss, (list, np.ndarray)):  
            loss = np.mean(loss)  # Convert list/array to a scalar
        elif hasattr(loss, "item"):  
            loss = loss.item()
        accuracy = compute_accuracy(np.argmax(y_train, axis=1), np.argmax(nn.forward(x_train), axis=1))


        # Validate the model
        y_pred_val = nn.forward(x_val)
        val_loss = nn.compute_loss(y_val, y_pred_val)
        val_accuracy = compute_accuracy(np.argmax(y_val, axis=1), np.argmax(y_pred_val, axis=1))


        # Update best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy


        # Log the results
        wandb.log({
            "epoch": epoch + 1,
            "loss": loss,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "best_val_accuracy": best_val_accuracy  # Track best validation accuracy
            })


    run.finish()  # Ensures WandB properly ends the run


# Run the sweep
wandb.agent(sweep_id, function=train, count=1)


# Initialize wandb run (modify project name as needed)
wandb.init(project="fashion_mnist_sweep", name="best_model_evaluation")


# BEST Neural network configuration.
input_dim = 28 * 28  # 784 features.
output_dim = 10 # 10 classes.
num_hidden_layers = 3
hidden_layer_dim = 128
learning_rate = 0.001
epochs = 10  # You may increase epochs for better performance.
batch_size = 32  # Mini-batch size.
weight_init = "random"
weight_decay = 0.5


# Initialize the neural network.
nn = FeedForwardNN(input_dim, output_dim, num_hidden_layers, hidden_layer_dim,
                       lr=learning_rate, act_hidden="relu", act_output="softmax", weight_init = "random", weight_decay = 0.0)


# Select optimizer (options: "sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam").
optimizer_type = "rmsprop"
optimizer = Optimizer(parameters=nn.W, optimizer_type=optimizer_type, lr=learning_rate)


# Train the network with mini-batch training.
print("Training started with mini-batch training...\n")
# Load and Preprocess Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# Normalize pixel values (0 to 1)
x_train, x_test = x_train / 255.0, x_test / 255.0


# Flatten images for neural network input
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)


# One-hot encode labels
num_classes = 10
y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]
loss_history = nn.train(x_train, y_train_onehot, epochs=epochs, batch_size=batch_size, print_every=1, optimizer=optimizer)


# Evaluate on training data.
Y_pred_test = nn.forward(x_test)
test_predictions = np.argmax(Y_pred_test, axis=1)
test_accuracy = compute_accuracy(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.2%}")


# Compute Confusion Matrix using NumPy
# Initialize the W&B run
wandb.init(project="fashion-mnist-classification", name="confusion_matrix_sweep")
num_classes = 10
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for true, pred in zip(y_test, test_predictions):
    conf_matrix[true, pred] += 1

# Define class labels
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Create confusion matrix dictionary
conf_matrix_dict = {
    "y_true": [],
    "y_pred": [],
    "count": []
}

for i in range(num_classes):
    for j in range(num_classes):
        if conf_matrix[i][j] > 0:
            conf_matrix_dict["y_true"].append(class_labels[i])
            conf_matrix_dict["y_pred"].append(class_labels[j])
            conf_matrix_dict["count"].append(conf_matrix[i][j])

# Log using `wandb.plot.confusion_matrix`
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=[class_labels[i] for i in y_test],
        preds=[class_labels[i] for i in test_predictions],
        class_names=class_labels
    )
})

# Finish wandb run
wandb.finish()

import matplotlib.pyplot as plt

class FeedForwardNN:
    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_layer_dim, lr=0.01,
                 act_hidden="sigmoid", act_output="softmax", weight_init="random", weight_decay=0.0):
        self.lr = lr
        self.act_hidden = act_hidden
        self.act_output = act_output
        self.weight_init = weight_init
        self.num_hidden_layers = num_hidden_layers
        self.layer_dims = [input_dim] + [hidden_layer_dim] * num_hidden_layers + [output_dim]
        self.num_layers = len(self.layer_dims) - 1
        self.W = self.initialize_weights(input_dim, hidden_layer_dim, output_dim)

    def initialize_weights(self, input_dim, hidden_dim, output_dim):
        W = []
        layer_sizes = [input_dim] + [hidden_dim] * self.num_hidden_layers + [output_dim]
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i] + 1
            output_size = layer_sizes[i + 1]
            if self.weight_init == "xavier":
                limit = np.sqrt(6 / (input_size + output_size))
                W.append(np.random.uniform(-limit, limit, (input_size, output_size)))
            else:
                W.append(np.random.randn(input_size, output_size) * 0.01)
        return W

    def forward(self, X):
        n_samples = X.shape[0]
        self.a_list = []
        self.z_list = []
        a = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        self.a_list.append(a)

        for i in range(self.num_layers):
            z = a @ self.W[i]
            self.z_list.append(z)
            if i < self.num_layers - 1:
                a = activate(z, self.act_hidden)
                a = np.concatenate([a, np.ones((n_samples, 1))], axis=1)
            else:
                a = activate(z, self.act_output)
            self.a_list.append(a)

        return a
    
    def compute_loss(self, Y_pred, Y_true, loss_type="cross_entropy", weight_decay=0.0):
        if loss_type == "cross_entropy":
            loss = -np.mean(np.sum(Y_true * np.log(Y_pred + 1e-8), axis=1))
        elif loss_type == "squared_error":
            loss = 0.5 * np.mean(np.sum((Y_true - Y_pred) ** 2, axis=1))
        else:
            raise ValueError("Unsupported loss type: " + loss_type)

        # L2 Regularization (Weight Decay)
        if weight_decay > 0:
            l2_penalty = sum(np.sum(W ** 2) for W in self.W) * (weight_decay / 2)
            loss += l2_penalty

        return loss
    
    def backward(self, X, Y_true, loss_type="cross_entropy", weight_decay=0.0):
        grads = [None] * self.num_layers
        n_samples = X.shape[0]
        Y_pred = self.a_list[-1]

        for i in reversed(range(self.num_layers)):
            z = self.z_list[i]
            a_prev = self.a_list[i]
            if i == self.num_layers - 1:
                if loss_type == "cross_entropy":
                    dZ = (Y_pred - Y_true) / n_samples
                elif loss_type == "squared_error":
                    dZ = (Y_pred - Y_true) * activation_derivative(z, self.act_output)
            else:
                dZ = dA * activation_derivative(z, self.act_hidden)

            dW = a_prev.T @ dZ
            if weight_decay > 0:
                dW += weight_decay * self.W[i]
            grads[i] = dW

            dA = dZ @ self.W[i].T
            dA = dA[:, :-1]

        return grads

    def update_params(self, grads):
        for i in range(self.num_layers):
            self.W[i] -= self.lr * grads[i]

    def train(self, X, Y, epochs=20, batch_size=None, loss_type="cross_entropy"):
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples

        loss_history = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            epoch_loss = 0.0
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                Y_pred = self.forward(X_batch)
                loss = self.compute_loss(Y_pred, Y_batch, loss_type)
                epoch_loss += loss * (X_batch.shape[0] / n_samples)

                grads = self.backward(X_batch, Y_batch, loss_type)
                self.update_params(grads)

            loss_history.append(epoch_loss)
            wandb.log({f"{loss_type}_loss": epoch_loss})

            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | {loss_type} Loss: {epoch_loss:.4f}")

        return loss_history

# ----------------------------------------
# Initialize W&B
wandb.init(project="fashion-mnist-classification", name="loss-comparison")

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
y_train = np.eye(10)[y_train]

# Train using cross entropy loss
nn_cross_entropy = FeedForwardNN(input_dim=28*28, output_dim=10,
                                 num_hidden_layers=2, hidden_layer_dim=128,
                                 lr=0.01)
cross_entropy_loss = nn_cross_entropy.train(x_train, y_train, epochs=20, loss_type="cross_entropy")

# Train using squared error loss
nn_squared_error = FeedForwardNN(input_dim=28*28, output_dim=10,
                                 num_hidden_layers=2, hidden_layer_dim=128,
                                 lr=0.01)
squared_error_loss = nn_squared_error.train(x_train, y_train, epochs=20, loss_type="squared_error")

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(cross_entropy_loss, label='Cross Entropy Loss', marker='o')
plt.plot(squared_error_loss, label='Squared Error Loss', marker='o')
plt.title('Loss Comparison: Cross Entropy vs Squared Error')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Log to W&B
wandb.log({"loss_comparison": wandb.Image(plt)})

# Finish W&B
wandb.finish()

from keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Define configurations
configs = [
    {
        "name": "config_1",
        "learning_rate": 0.001,
        "num_hidden_layers": 3,
        "hidden_layer_size": 128,
        "optimizer": "adam",
        "activation": "relu",
        "batch_size": 128,
        "epochs": 10,
        "weight_init": "xavier",        # Options: 'random', 'xavier'
        "weight_decay": 1e-4        # L2 regularization
    },
    {
        "name": "config_2",
        "learning_rate": 0.0005,
        "num_hidden_layers": 4,
        "hidden_layer_size": 256,
        "optimizer": "rmsprop",
        "activation": "relu",
        "batch_size": 128,
        "epochs": 10,
        "weight_init": "xavier",
        "weight_decay": 1e-5
    },
    {
        "name": "config_3",
        "learning_rate": 0.01,
        "num_hidden_layers": 3,
        "hidden_layer_size": 64,
        "optimizer": "momentum",
        "activation": "relu",
        "batch_size": 128,
        "epochs": 10,
        "weight_init": "random",
        "weight_decay": 1e-3
    }
]

# Loop through configurations
results = []
for config in configs:
    print(f"\n=== Training with {config['name']} ===\n")

    # Initialize wandb for logging
    wandb.init(project="fashion-mnist-classification", name=config["name"], config=config)
    
    # Initialize the neural network with weight initialization and decay
    nn = FeedForwardNN(
        input_dim=28 * 28,
        output_dim=10,
        num_hidden_layers=config["num_hidden_layers"],
        hidden_layer_dim=config["hidden_layer_size"],
        lr=config["learning_rate"],
        act_hidden=config["activation"],
        act_output="softmax",
        weight_init=config["weight_init"],    # NEW
        weight_decay=config["weight_decay"]   # NEW
    )

    # Select optimizer
    optimizer = Optimizer(
        parameters=nn.W,
        optimizer_type=config["optimizer"],
        lr=config["learning_rate"]
    )

    # Train the network
    loss_history = nn.train(
        x_train, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        optimizer=optimizer
    )

    # Evaluate on test set
    Y_pred_test = nn.forward(x_test)
    test_predictions = np.argmax(Y_pred_test, axis=1)
    test_accuracy = compute_accuracy(np.argmax(y_test, axis=1), test_predictions)

    # Log accuracy
    wandb.log({"test_accuracy": test_accuracy})
    print(f"Test Accuracy for {config['name']}: {test_accuracy:.2%}")

    # Confusion Matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(np.argmax(y_test, axis=1), test_predictions):
        conf_matrix[true][pred] += 1

    class_labels = [
        "0", "1", "2", "3", "4",
        "5", "6", "7", "8", "9"
    ]

    # Log confusion matrix
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=[class_labels[i] for i in np.argmax(y_test, axis=1)],
            preds=[class_labels[i] for i in test_predictions],
            class_names=class_labels
        )
    })

    # Save results
    results.append({
        "name": config["name"],
        "test_accuracy": test_accuracy
    })

    # Finish run
    wandb.finish()

# Report results
print("\n=== Final Results ===")
for result in results:
    print(f"{result['name']}: {result['test_accuracy']:.2%}")



