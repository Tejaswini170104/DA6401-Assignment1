# DA6401-Assignment1
wandb report link : https://wandb.ai/tejaswiniksssn-indian-institute-of-technology-madras/fashion-mnist-classification-v7/reports/DA6401-asg01-report--VmlldzoxMTgzOTg5NQ <br>
github repo link : https://github.com/Tejaswini170104/DA6401-Assignment1 <br>
# Question 1: <br>
Objective: <br>
Log one sample image per class from the Fashion-MNIST dataset to Weights & Biases (W&B) for visualization.

# Question 2 & Question 3: <br>
This project implements a feedforward neural network from scratch for classifying the Fashion-MNIST dataset. It includes flexible network architecture, backpropagation, and support for multiple optimizers. <br>

Activation Functions & Derivatives: <br>
relu(x) – Applies ReLU activation (sets negative values to zero).<br>
relu_derivative(x) – Derivative of ReLU (1 if x > 0, else 0).<br>
tanh(x) – Applies the Tanh activation.<br>
tanh_derivative(x) – Derivative of Tanh.<br>
sigmoid(x) – Applies the Sigmoid activation.<br>
sigmoid_derivative(x) – Derivative of Sigmoid.<br>
softmax(x) – Applies the Softmax function for multi-class classification.<br>

Activation Utilities: <br>
activate(x, act_type) – Applies the specified activation function.<br>
activation_derivative(x, act_type) – Computes the derivative of the specified activation function.<br>

Compute Accuracy: <br>
compute_accuracy(y_true, y_pred) – Calculates the classification accuracy.<br>

FeedForwardNN Class: <br>
__init__() – Initializes the neural network architecture and weights.<br>
initialize_weights() – Initializes weights using Xavier or random initialization.<br>
forward() – Forward pass through the network layers.<br>
compute_loss() – Computes loss (cross-entropy or squared error) with optional weight decay.<br>
backward() – Backpropagation to calculate gradients.<br>
default optimizer set as the best performing one, "adam". <br>
train() – Trains the neural network with mini-batches and specified optimizer.<br>

Optimizer Class: <br>
__init__() – Initializes the optimizer parameters.<br>
step() – Updates parameters using specified optimization algorithm:<br>
sgd – Simple Stochastic Gradient Descent.<br>
momentum – SGD with momentum for faster convergence.<br>
nesterov – Nesterov accelerated gradient descent.<br>
rmsprop – Root Mean Square Propagation.<br>
adam – Adaptive Moment Estimation.<br>
nadam – Nesterov-accelerated Adam.<br>

Example Usage: <br>
Dataset – Fashion-MNIST dataset with 60,000 training and 10,000 test images, each of size 28x28.<br>
Input Dimension – 784 (28x28 flattened).<br>
Output Classes – 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot).<br>
Hidden Layers – 2 hidden layers with 128 neurons each.<br>
Learning Rate – 0.01.<br>
Batch Size – 128 (Mini-Batch Gradient Descent).<br>
Epochs – 10.<br>
Weight Initialization – Random.<br>
Activation Functions – Sigmoid for hidden layers, Softmax for output layer.<br>
Optimizer – Adam.<br>
Loss Type – Cross Entropy Loss.<br>
Weight Decay – 0.0 (No L2 Regularization).<br>

# Question 4: <br>
Sweep Example Explanation: answers in wandb report. <br>
Sweep Config: Used bayes search to maximize val_accuracy. <br>
Data Handling: Loaded Fashion MNIST, reshaped to vectors, normalized, and one-hot encoded labels. <br>
Model Setup: Created FeedForwardNN with hyperparameters from the sweep config. <br>
Training: Forward + Backward pass using the selected optimizer and hyperparameters. <br>
Logging: Recorded loss, accuracy, validation loss, and best validation accuracy to W&B. <br>
Execution: Ran sweep with wandb.agent() for 30 combinations, set run names for clarity. <br>

# Question 5: <br>
Best Neural Network Configuration: <br>
Input Dim: 784, Output Dim: 10 (10 classes) <br>
Hidden Layers: 5, Hidden Layer Size: 128 <br>
Learning Rate: 0.0001, Weight Initialization: Xavier <br>
Optimizer: Adam, Activation: ReLU, Batch Size: 16 <br>
Epochs: 20, Weight Decay: 0 <br>
Performance: <br>
Achieved highest validation accuracy with the above configuration.
Achieved test accuracy = 88.62%  <br>

# Question 6: <br>
Inferences and observations in wandb report. <br>

# Question 7: <br>
Confusion Matrix for Best Model: <br>
Generated a creative heatmap of the confusion matrix using Seaborn for its clear color mapping and annotation support. <br>
Logged the matrix as a W&B artifact for easy tracking and visualization. <br>

# Question 8: <br>
Loss Comparison: Cross Entropy vs Squared Error : <br>
Compared Cross Entropy Loss with Squared Error Loss using the same network structure and hyperparameters.<br>
Cross entropy is generally better for classification tasks as it directly optimizes the probability distribution.<br>
Trained two identical networks with different loss functions and logged the loss curves using W&B.<br>
Logged loss curves directly to W&B as a line plot to visualize performance differences over epochs.<br>

# Question 10: <br>
MNIST Hyperparameter Recommendations: <br>
Based on experimentation with Fashion-MNIST, the following 3 hyperparameter configurations were tested on the MNIST dataset:<br>

Config 1 : 5 hidden layers (128), learning rate = 0.0001, optimizer = Adam, activation = ReLU, batch size = 16, weight decay = 0 → Achieved test accuracy 97.65 %.<br>
Config 2 : 4 hidden layers (128), learning rate = 0.0005, optimizer = RMSprop, activation = Sigmoid, batch size = 32, weight decay = 1e-5 → Achieved test accuracy 96.95 %.<br>
Config 3 : 3 hidden layers (128), learning rate = 0.001, optimizer = Nadam, activation = ReLU, batch size = 64, weight decay = 0.0005 → Achieved test accuracy 97.92 %.<br>



