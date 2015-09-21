package neural

/* Structs */

type NeuralNetwork struct {
	layers         []Layer
	activationFunc func(float64) float64
}

type Layer struct {
	neurons []Neuron
}

type Neuron struct {
	bias    float64
	weights []float64
}

/* NeuralNetwork methods */

func (nn NeuralNetwork) Activate(inputValues []float64) []float64 {
	var outputs []float64 = inputValues

	for i, layer := range nn.layers {
		// Skip the input layer
		if i == 0 {
			continue
		}

		outputs = layer.computeOutputs(outputs, nn.activationFunc)
	}

	// The last value for inputs is the output of the output nodes
	return outputs
}

func (nn *NeuralNetwork) SetConnectionWeight(layer, input, index int, weight float64) {
	nn.layers[layer].neurons[index].weights[input] = weight
}

func (nn *NeuralNetwork) SetNeuronWeights(layer, index int, weights []float64) {
	nn.layers[layer].neurons[index].weights = weights
}

func (nn *NeuralNetwork) SetNeuronBias(layer, index int, bias float64) {
	nn.layers[layer].neurons[index].bias = bias
}

func (nn *NeuralNetwork) SetDefaultBias(bias float64) {
	for i, layer := range nn.layers {
		if i == 0 {
			continue
		}
		for j := range layer.neurons {
			nn.layers[i].neurons[j].bias = bias
		}
	}
}

/* Layer methods */

func (l Layer) computeOutputs(inputs []float64, activate func(float64) float64) []float64 {
	outputs := make([]float64, len(l.neurons))
	for i, neuron := range l.neurons {
		sum := neuron.sum(inputs)
		outputs[i] = activate(sum)
	}
	return outputs
}

/* Neuron methods */

func (n Neuron) sum(inputs []float64) float64 {
	sum := 0.0
	for i, weight := range n.weights {
		sum += inputs[i] * weight
	}
	sum += n.bias
	return sum
}

/* Constructor */

func CreateNeuralNetwork(shape []int, activationFunc func(float64) float64) NeuralNetwork {
	layers := make([]Layer, len(shape))
	outputSize := shape[0]

	for i, size := range shape {
		neurons := make([]Neuron, size)
		for j := range neurons {
			neurons[j].weights = make([]float64, outputSize)
		}

		layers[i] = Layer{neurons}
		outputSize = size
	}

	return NeuralNetwork{layers, activationFunc}
}

/* Helper functions */

func max(values []int) int {
	maxInt := 0
	for _, v := range values {
		if v > maxInt {
			maxInt = v
		}
	}
	return maxInt
}
