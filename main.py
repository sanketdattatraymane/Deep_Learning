import numpy as np

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid Activation Function
def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network Class Definition
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases with random values
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.randn(1, self.hidden_size)

        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.random.randn(1, self.output_size)

    def forward(self, X):
        self.input_layer = X
        self.hidden_layer_input = np.dot(self.input_layer, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = sigmoid(self.output_layer_input)

        return self.output_layer_output

    def backward(self, X, y, learning_rate):
        error_output = y - self.output_layer_output
        output_layer_delta = error_output * sigmoid_derivative(self.output_layer_output)

        error_hidden = output_layer_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = error_hidden * sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_layer_delta) * learning_rate
        self.bias_output += np.sum(output_layer_delta, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden += X.T.dot(hidden_layer_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.output_layer_output))  # Mean squared error
                print(f"Epoch {epoch} - Loss: {loss}")

# Main Program
if __name__ == "__main__":
    # Take user input for the network configuration
    input_size = int(input("Enter the number of input features: "))
    hidden_size = int(input("Enter the number of hidden neurons: "))
    output_size = int(input("Enter the number of output neurons: "))

    # Use XOR dataset for testing (2 inputs, 1 output)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data (XOR inputs)
    y = np.array([[0], [1], [1], [0]])  # Expected output data (XOR outputs)

    # Ensure input_size matches the dataset's number of features
    if X.shape[1] != input_size:
        print(f"Error: The input size should match the number of features in the dataset (currently {X.shape[1]} features).")
        exit()

    # Create the neural network with the user-defined configuration
    nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    # Train the network for 10,000 epochs with a learning rate of 0.1
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    # After training, print the final predictions of the network
    print("\nPredictions after training:")
    print(nn.forward(X))  # Test the network on the XOR inputs

    # Allow the user to test the trained model with custom inputs
    print("\nEnter new inputs to test the model (type 'exit' to quit):")
    while True:
        user_input = input(f"Input {input_size} comma-separated values: ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        try:
            user_data = np.array([list(map(float, user_input.split(",")))])
            if user_data.shape[1] != input_size:
                print(f"Error: Please enter exactly {input_size} values!")
                continue
            prediction = nn.forward(user_data)
            print(f"Model Output: {prediction}")
        except ValueError:
            print("Invalid input. Please enter numerical values separated by commas.")
