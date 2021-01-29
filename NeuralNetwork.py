from Matrix import Matrix
from utility import sigmoid, d_sigmoid


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_ih = Matrix(hidden_nodes, input_nodes)
        self.weights_ho = Matrix(output_nodes, hidden_nodes)
        self.weights_ih.randomize()
        self.weights_ho.randomize()

        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)
        self.bias_h.randomize()
        self.bias_o.randomize()

        self.learning_rate = learning_rate

    # Return predicted output without training the network
    def feed_forward(self, input_array):
        inputs = Matrix.from_array(input_array)

        hidden = Matrix.multiply(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        # Activation function
        hidden.map(sigmoid)

        output = Matrix.multiply(self.weights_ho, hidden)
        output.add(self.bias_o)
        # Activation function
        output.map(sigmoid)

        return Matrix.to_array(output)

    def get_weights(self):
        state = {
            'input-hidden': self.weights_ih,
            'hidden-output': self.weights_ho
        }
        return state

    def get_activations(self):
        state = {
            'input': self.inputs,
            'hidden': self.hidden,
            'output': self.outputs
        }
        return state

    def train(self, input_array, target_array):
        inputs = Matrix.from_array(input_array)
        self.inputs = inputs
        # Activations of Hidden Layer
        hidden = Matrix.multiply(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        # Activation function
        hidden.map(sigmoid)
        self.hidden = hidden
        # Activations of Output layer
        outputs = Matrix.multiply(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        # Activation function
        outputs.map(sigmoid)
        self.outputs = outputs

        targets = Matrix.from_array(target_array)
        # Calculate the error
        # OUTPUT_ERROR = TARGETS - OUTPUTS
        output_errors = Matrix.subtract(targets, outputs)

        # Calculate gradient
        # GRADIENT = OUTPUTS * (1 - OUTPUTS) * LEARNING_RATE
        gradients = Matrix.map_static(outputs, d_sigmoid)
        gradients.scale(output_errors)
        gradients.scale(self.learning_rate)

        # Calculate hidden to output deltas
        # DELTA = GRADIENTS * HIDDEN_t
        hidden_t = Matrix.transpose(hidden)
        weights_ho_delta = Matrix.multiply(gradients, hidden_t)

        # Adjust bias
        self.bias_o.add(gradients)

        # Calculate hidden errors
        # HIDDEN_ERROR = HIDDEN_WEIGHTS_t *  OUTPUT_ERROR
        weights_ho_t = Matrix.transpose(self.weights_ho)
        hidden_errors = Matrix.multiply(weights_ho_t, output_errors)

        # Calculate hidden gradient
        hidden_gradients = Matrix.map_static(hidden, d_sigmoid)
        hidden_gradients.scale(hidden_errors)
        hidden_gradients.scale(self.learning_rate)

        # Calculate input to hidden deltas
        # HIDDEN_DELTA = GRADIENTS * INPUT_t
        inputs_t = Matrix.transpose(inputs)
        weights_ih_delta = Matrix.multiply(hidden_gradients, inputs_t)

        self.bias_h.add(hidden_gradients)

        self.weights_ih.add(weights_ih_delta)
        self.weights_ho.add(weights_ho_delta)

